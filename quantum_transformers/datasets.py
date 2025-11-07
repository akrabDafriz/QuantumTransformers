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
                        data_dir=None, tokenizer=None): 
    """
    Downloads and prepares a dataset for Masked Language Modeling using an efficient chunking strategy.
    Uses dataset slicing to avoid OOM errors.
    """
    
    print("Loading dataset slices from Hugging Face (to save RAM)...")
    # Load slices of the dataset to prevent OOM errors
    # We use a combined slice for programmatic splitting
    # We take 200k for train, 10k for val, 10k for test
    raw_datasets_train = load_dataset(
        dataset_name, 
        split='train[:200000]', # Load 200k chunk for training
        cache_dir=data_dir
    )
    raw_datasets_val = load_dataset(
        dataset_name,
        split='train[200000:210000]', # Load 10k chunk for validation
        cache_dir=data_dir
    )
    raw_datasets_test = load_dataset(
        dataset_name,
        split='train[210000:220000]', # Load 10k chunk for testing
        cache_dir=data_dir
    )
    
    print("Dataset slices loaded.")

    # Load tokenizer
    print("Loading tokenizer...")
    # --- MODIFICATION: Use the passed-in tokenizer if it exists ---
    if tokenizer is not None:
        print("Using tokenizer provided from training script.")
        print(tokenizer)
    else:
        # Fallback to AutoTokenizer if no tokenizer is passed
        print(f"Loading tokenizer from model checkpoint: {model_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    # --- END MODIFICATION ---


    def tokenize_function(examples):
        """
        Tokenizes text. 
        --- FIX: REMOVED padding and truncation. ---
        The group_texts function needs raw, un-padded token lists.
        """
        print("Tokenizing batch...")
        # Print the first text to debug
        if examples and 'text' in examples and examples['text']:
             print(f"First text in batch: {examples['text'][0][:50]}...")
        
        # --- THIS IS THE FIX ---
        # Do not pad or truncate here. Just tokenize.
        # The 'group_texts' function will handle the chunking.
        return tokenizer(examples["text"])
        # --- END FIX ---

    print("\nTokenizing datasets...")
    # We map the tokenize_function to our dataset slices
    temp_tokenized_datasets = {
        'train': raw_datasets_train.map(tokenize_function, batched=True, remove_columns=["text"]),
        'validation': raw_datasets_val.map(tokenize_function, batched=True, remove_columns=["text"]),
        'test': raw_datasets_test.map(tokenize_function, batched=True, remove_columns=["text"])
    }
    print("Tokenization complete.")

    # --- DEBUG: Print an example from the *tokenized* (but not grouped) set ---
    print("\nExample from tokenized (pre-grouping) test split:")
    print(temp_tokenized_datasets['test'][0])
    # --- END DEBUG ---

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
        result["labels"] = result["input_ids"].copy()
        return result

    print("\nGrouping texts into blocks...")
    # Get column names to remove *after* tokenization
    column_names = list(temp_tokenized_datasets['train'].features)
    print(f"Column names before grouping:\n{column_names}")
    
    tokenized_datasets = {
        'train': temp_tokenized_datasets['train'].map(
            group_texts, 
            batched=True,
            remove_columns=column_names # Remove old columns
        ),
        'validation': temp_tokenized_datasets['validation'].map(
            group_texts, 
            batched=True,
            remove_columns=column_names # Remove old columns
        ),
        'test': temp_tokenized_datasets['test'].map(
            group_texts, 
            batched=True,
            remove_columns=column_names # Remove old columns
        )
    }
    print("Text grouping complete.")
    
    # --- DEBUG: Print column names *after* grouping ---
    print(f"Column names after grouping:\n{list(tokenized_datasets['train'].features)}")
    # --- END DEBUG ---
    
    # --- DEBUG: Print an example from the *grouped* set ---
    print("\nExample from grouped train split:")
    print(tokenized_datasets['train'][0])
    # --- END DEBUG ---

    # Use PyTorch dataloaders for easier integration with DataCollator
    print("\nSetting format to 'torch' for dataloaders...")
    tokenized_datasets['train'].set_format("torch")
    tokenized_datasets['validation'].set_format("torch")
    tokenized_datasets['test'].set_format("torch")
    
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"train split size: {len(train_dataset)}")
    print(f"validation split size: {len(val_dataset)}")
    print(f"test split size: {len(test_dataset)}")

    # Data collator will take care of creating MLM inputs and labels
    print("Initializing Data Collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
    print("\nData loaders created successfully.")

    # Convert PyTorch dataloaders to callables that create fresh generators
    def create_numpy_dataloader(dataloader):
        def generator():
            for batch in dataloader:
                yield (batch['input_ids'].numpy(), batch['labels'].numpy())
        return generator
            
    # --- DEBUG: Print a batch to check ---
    print("\nExample batch from train dataloader:")
    example_batch = next(iter(train_dataloader))
    print("Input IDs:")
    print(example_batch['input_ids'])
    print("Labels:")
    print(example_batch['labels'])
    # --- END DEBUG ---

    return (create_numpy_dataloader(train_dataloader),
            create_numpy_dataloader(val_dataloader),
            create_numpy_dataloader(test_dataloader)), tokenizer


# --- THIS IS THE OLD IMDB FUNCTION, WE LEAVE IT HERE ---
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