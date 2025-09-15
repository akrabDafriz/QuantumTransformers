import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_text import WordpieceTokenizer

# Imports for MLM data handling
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling # This library is provided by Hugging Face
import torch


def get_dataloaders(dataset_name, batch_size, data_dir=None, **kwargs):
    if dataset_name == "imdb":
        return get_imdb_dataloaders(batch_size, data_dir, **kwargs)
    elif dataset_name == "mlm":
        return get_mlm_dataloaders(batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_imdb_dataloaders(batch_size, data_dir, max_vocab_size=20_000, max_seq_len=512):
    (ds_train, ds_val, ds_test), ds_info = tfds.load('imdb_reviews', split=['train[:90%]', 'train[90%:]', 'test'],
                                                    data_dir=data_dir, as_supervised=True, with_info=True)
    
    print(f"Cardinalities (train, val, test): {ds_train.cardinality().numpy()} {ds_val.cardinality().numpy()} {ds_test.cardinality().numpy()}")
    
    # Vocabulary and tokenizer
    # Note that this is not a great way of building a vocabulary since it only uses the training set,
    # and it might also be slow. It is however convenient for a quick-and-dirty implementation.
    vocab = set()
    tokenizer = WordpieceTokenizer(vocab)
    for review, _ in ds_train.take(ds_train.cardinality()):
        tokens = tokenizer.tokenize(review)
        vocab.update(tokens.numpy().tolist())
    
    # Sort the vocabulary and limit its size
    vocab = sorted(list(vocab))
    vocab = vocab[:max_vocab_size]
    
    # Create the final tokenizer
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


# Updated function for MLM data
def get_mlm_dataloaders(batch_size, dataset_name='Helsinki-NLP/opus_books', dataset_config_name='en-es',
                        model_checkpoint='bert-base-uncased', block_size=128, mlm_probability=0.15):
    """
    Downloads and prepares a dataset for Masked Language Modeling using an efficient chunking strategy.
    Defaults to the Helsinki-NLP/opus_books dataset.
    """
    # Load dataset from Hugging Face
    raw_datasets = load_dataset(dataset_name, dataset_config_name, split='train')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        # Extract the English text from the 'translation' column
        english_texts = [ex['en'] for ex in examples['translation']]
        # Tokenize without padding
        return tokenizer(english_texts)

    # The opus_books dataset doesn't have standard train/val/test splits, so we'll create them.
    train_test_split = raw_datasets.train_test_split(test_size=0.1)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1)
    
    temp_tokenized_datasets = {
        'train': train_val_split['train'].map(tokenize_function, batched=True, remove_columns=["id", "translation"]),
        'validation': train_val_split['test'].map(tokenize_function, batched=True, remove_columns=["id", "translation"]),
        'test': train_test_split['test'].map(tokenize_function, batched=True, remove_columns=["id", "translation"])
    }

    # Function to group texts into chunks of block_size so that we can train the model on fixed-length sequences instead of padding each individual example to block_size
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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
    # This datacollator will dynamically mask tokens in the input_ids and set the labels accordingly.

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    # Convert PyTorch dataloaders to callables that create fresh generators
    def create_numpy_dataloader(dataloader):
        def generator():
            for batch in dataloader:
                # The collator returns a dict of torch tensors.
                # We need to yield `(inputs, labels)`.
                # For MLM, the `labels` are the input_ids with non-masked tokens set to -100.
                # The model input should be the `input_ids` from the batch.
                yield (batch['input_ids'].numpy(), batch['labels'].numpy())
        return generator
            
    # After finding out that the Transformer model expects jnp arrays, we might want to convert the torch tensors to jnp arrays here instead of numpy arrays.
    # However, the dataloader itself yields torch tensors, so we need to convert them to numpy arrays first. Then we can convert them to jnp arrays in the training loop.
    # Alternatively, we could write a custom dataloader that yields jnp arrays directly, but that might be more complex.
    # For now, we will keep it as numpy arrays and convert to jnp arrays in the training loop.
    
    # Return callables that create fresh generators each time
    return (create_numpy_dataloader(train_dataloader),
            create_numpy_dataloader(val_dataloader),
            create_numpy_dataloader(test_dataloader)), tokenizer