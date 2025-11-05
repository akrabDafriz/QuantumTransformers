# --- 1. FRAMEWORK SETUP (MUST BE FIRST) ---
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
# This MUST run before JAX is imported.
tf.config.set_visible_devices([], device_type='GPU')
# --- END FRAMEWORK SETUP ---

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# --- THIS IS THE MAIN FUNCTION CALLED BY THE TRAINING SCRIPT ---
def get_mlm_dataloaders(batch_size, dataset_name='roneneldan/TinyStories',
                        model_checkpoint='bert-base-uncased', block_size=128, mlm_probability=0.15,
                        data_dir=None):
    """
    Downloads and prepares a dataset for Masked Language Modeling using an efficient chunking strategy.
    Uses dataset slicing to avoid OOM errors.
    """
    
    print("Loading dataset slices from Hugging Face (to save RAM)...")
    # Load slices of the dataset to prevent OOM errors
    # We use a combined slice for programmatic splitting
    raw_datasets = load_dataset(
        dataset_name, 
        split='train[:200000]', # Load one 200k chunk
        cache_dir=data_dir
    )
    print("Dataset slices loaded.")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    
    def tokenize_function(examples):
        # Tokenize the 'text' column
        return tokenizer(examples['text'])

    print("Splitting dataset...")
    # Programmatically split the 200k chunk into train/val/test
    # 90% train (180k), 10% test (20k)
    train_test_split = raw_datasets.train_test_split(test_size=0.1, seed=42)
    # Of the 180k train, 90% train (162k), 10% validation (18k)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)

    temp_tokenized_datasets = {
        'train': train_val_split['train'].map(
            tokenize_function, batched=True, remove_columns=["text"]),
        'validation': train_val_split['test'].map(
            tokenize_function, batched=True, remove_columns=["text"]),
        'test': train_test_split['test'].map(
            tokenize_function, batched=True, remove_columns=["text"])
    }
    
    print(f"Train examples: {len(temp_tokenized_datasets['train'])}")
    print(f"Validation examples: {len(temp_tokenized_datasets['validation'])}")
    print(f"Test examples: {len(temp_tokenized_datasets['test'])}")


    # Function to group texts into chunks of block_size
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # --- THIS IS THE CRITICAL LINE THAT FIXES THE PPL=1.0 BUG ---
        # The data collator EXPECTS this. It will internally create the
        # correct -100 labels based on the masking it performs.
        result["labels"] = result["input_ids"].copy()
        # --- END CRITICAL LINE ---
        
        return result

    print("Grouping texts into blocks...")
    tokenized_datasets = {
        'train': temp_tokenized_datasets['train'].map(group_texts, batched=True),
        'validation': temp_tokenized_datasets['validation'].map(group_texts, batched=True),
        'test': temp_tokenized_datasets['test'].map(group_texts, batched=True)
    }

    # Use PyTorch dataloaders for easier integration with DataCollator
    tokenized_datasets['train'].set_format("torch")
    tokenized_datasets['validation'].set_format("torch")
    tokenized_datasets['test'].set_format("torch")
    
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # Data collator will take care of creating MLM inputs and labels
    print("Initializing Data Collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm_probability=mlm_probability
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=data_collator
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    # Convert PyTorch dataloaders to callables that create fresh generators
    def create_numpy_dataloader(dataloader):
        def generator():
            for batch in dataloader:
                # The collator returns a dict of torch tensors.
                # 'input_ids' are now MASKED
                # 'labels' are the original IDs, with -100 for non-masked tokens
                yield (batch['input_ids'].numpy(), batch['labels'].numpy())
        return generator
            
    # Return callables that create fresh generators each time
    return (create_numpy_dataloader(train_dataloader),
            create_numpy_dataloader(val_dataloader),
            create_numpy_dataloader(test_dataloader)), tokenizer


# --- THIS IS THE OLD IMDB FUNCTION, WE LEAVE IT HERE ---
# Note: It will not be called by mlm_training.py
import tensorflow_datasets as tfds
from tensorflow_text import WordpieceTokenizer

def get_imdb_dataloaders(batch_size, data_dir, max_vocab_size=20_000, max_seq_len=512):
    (ds_train, ds_val, ds_test), ds_info = tfds.load('imdb_reviews', split=['train[:90%]', 'train[90%:]', 'test'],
                                                    data_dir=data_dir, as_supervised=True, with_info=True)
    
    print(f"Cardinalities (train, val, test): {ds_train.cardinality().numpy()} {ds_val.cardinality().numpy()} {ds_test.cardinality().numpy()}")
    
    vocab = set()
    tokenizer = WordpieceTokenizer(vocab)
    for review, _ in ds_train.take(ds_train.cardinality()):
        tokens = tokenizer.tokenize(review)
        vocab.update(tokens.numpy().tolist())
    
    vocab = sorted(list(vocab))
    vocab = vocab[:max_vocab_size]
    
    tokenizer = WordpieceTokenizer(vocab, token_out_type=tf.int32)

    def encode(text, label):
        encoded_text = tokenizer.tokenize(text)
        return encoded_text, label
    
    def pad_and_batch(ds, batch_size):
        return ds.map(encode).padded_batch(batch_size, padded_shapes=([max_seq_len], []))

    ds_train = pad_and_batch(ds_train, batch_size)
    ds_val = pad_and_batch(ds_val, batch_size)
    ds_test = pad_and_batch(ds_test, batch_size)
    
    return (tfds.as_numpy(ds_train), tfds.as_numpy(ds_val), tfds.as_numpy(ds_test)), vocab, tokenizer