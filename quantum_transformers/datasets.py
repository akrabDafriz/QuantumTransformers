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
    raw_datasets = load_dataset(
        dataset_name, 
        split='train[:200000]', # Load one 200k chunk
        cache_dir=data_dir
    )
    val_dataset = load_dataset(dataset_name, split='validation', cache_dir=data_dir)

    print("Dataset slices loaded.")

    # --- Corrected code for datasets.py ---

    # Load tokenizer
    print("Loading tokenizer...")
    # --- MODIFICATION: Use the passed-in tokenizer if it exists! ---
    if tokenizer is None:
        print(f"No tokenizer provided, loading from checkpoint: {model_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    else:
        print("Using tokenizer provided from training script.")
        print(tokenizer)
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
        return tokenizer(examples["text"])

    print("Splitting dataset...")
    # Programmatically split the 200k chunk into train/val/test
    # 90% train (180k), 10% test (20k)
    train_test_split = raw_datasets.train_test_split(test_size=0.1, seed=42)
    # Of the 180k train, 90% train (162k), 10% validation (18k)
    # train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)

    # We removed `remove_columns` here, which was the correct fix.
    temp_tokenized_datasets = {
        'train': train_test_split['train'].map(
            tokenize_function, batched=True),
        'validation': val_dataset.map(
            tokenize_function, batched=True),
        'test': train_test_split['test'].map(
            tokenize_function, batched=True)
    }
    
    print(f"Train examples: {len(temp_tokenized_datasets['train'])}")
    print(f"Validation examples: {len(temp_tokenized_datasets['validation'])}")
    print(f"Test examples: {len(temp_tokenized_datasets['test'])}")

    # print 3 examples from each split
    for split in ['train', 'validation', 'test']:
        print(f"\nExamples from {split} split:")
        for i in range(3):
            print(temp_tokenized_datasets[split][i])


    # We update group_texts to ONLY concatenate the keys it knows about,
    # and to ignore the raw "text" column.
    def group_texts(examples):
        # Define the keys we *know* are tokenized lists
        keys_to_concatenate = ["input_ids", "token_type_ids", "attention_mask"]
        
        concatenated_examples = {}
        
        # Only sum the lists we care about
        for k in keys_to_concatenate:
            if k in examples: # Check if it exists (e.g., token_type_ids might not)
                concatenated_examples[k] = sum(examples[k], [])
        
        # Use the primary key (input_ids) to get the total length
        total_length = len(concatenated_examples["input_ids"])
        
        # We drop the small remainder.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split by chunks of block_size.
        result = {}
        for k, t in concatenated_examples.items():
            # Slice the concatenated list into chunks
            result[k] = [t[i : i + block_size] for i in range(0, total_length, block_size)]

        # This line is correct and necessary for the collator
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping texts into blocks...")
    
    column_names = temp_tokenized_datasets['train'].column_names 
    print(f"Column names: {column_names}")

    tokenized_datasets = {
            'train': temp_tokenized_datasets['train'].map(
            group_texts, 
            batched=True,
            remove_columns=column_names # <--- ADD THIS
        ),
        'validation': temp_tokenized_datasets['validation'].map(
            group_texts, 
            batched=True,
            remove_columns=column_names # <--- ADD THIS
        ),
        'test': temp_tokenized_datasets['test'].map(
            group_texts, 
            batched=True,
            remove_columns=column_names # <--- ADD THIS
        )
    }

    
    # --- END FIX ---
    # print 3 examples from each split after grouping
    for split in ['train', 'validation', 'test']:
        print(f"\nGrouped examples from {split} split:")
        for i in range(3):
            print(tokenized_datasets[split][i])

    # Print the column names of temp_tokenized_datasets
    print("\nColumn names before grouping:")
    for split in ['train', 'validation', 'test']:
        print(f"{split} split: {temp_tokenized_datasets[split].column_names}")
    
    # print the column names of the tokenized_datasets
    print("\nColumn names after grouping:")
    for split in ['train', 'validation', 'test']:
        print(f"{split} split: {tokenized_datasets[split].column_names}")
        
    # print the number of vocab size
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    
    # print the number of examples in each split
    for split in ['train', 'validation', 'test']:
        print(f"{split} split size: {len(tokenized_datasets[split])}")

    # Use PyTorch dataloaders for easier integration with DataCollator
    tokenized_datasets['train'].set_format("torch")
    tokenized_datasets['validation'].set_format("torch")
    tokenized_datasets['test'].set_format("torch")
    
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # Data collator will take care of creating MLM inputs and labels
    print("Initializing Data Collator...")
    
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id. This is fatal.")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm_probability=mlm_probability,
        mlm=True
    )
    data_collator.mask_token_id = tokenizer.mask_token_id


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
                yield (batch['input_ids'].numpy(), batch['labels'].numpy())
        return generator

    print("\nData loaders created successfully.")

    train_data = create_numpy_dataloader(train_dataloader)
    val_data = create_numpy_dataloader(val_dataloader)
    test_data = create_numpy_dataloader(test_dataloader)
    
    
    # Example of a masked batch
    for batch in train_dataloader:
        print("\nExample batch from train dataloader:")
        print("Input IDs:\n", batch['input_ids'])
        print("Labels:\n", batch['labels'])
        break
    
    # print the masking token
    print(f"\nMasking token ID: {tokenizer.mask_token_id}, Masking token: '{tokenizer.mask_token}'")

    # Return callables that create fresh generators each time
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