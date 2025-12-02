import os
import jax.numpy as jnp
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset
import torch

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode text
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        # Pad or truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [0] * (self.max_len - len(ids))

        return jnp.array(ids), jnp.array(label)


def train_tokenizer(texts, vocab_size=20000):
    """Trains a BPE tokenizer on the provided texts."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def load_mc_rp_data(file_path):
    """
    Parses MC (Meaning Classification) and RP (Relative Pronouns) datasets.
    Format: "1 organization_N that_RPRON fleet_N destroy_V" (Label first)
    """
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # The first token is the label (0 or 1)
            label = int(parts[0])
            # The rest is the sentence
            text = " ".join(parts[1:])
            
            labels.append(label)
            texts.append(text)
    return texts, labels


def load_sentiment_data(file_path):
    """
    Parses IMDb, Amazon, Yelp sentiment datasets.
    Format: "A bit predictable. \t 0" (Label last, tab or space separated)
    """
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # These files usually use tab \t or 4 spaces to separate sentence and label
            if '\t' in line:
                parts = line.split('\t')
                text = parts[0].strip()
                label = int(parts[-1])
            else:
                # Fallback: assume the last character is the label if no tab found
                # (Some versions of this dataset use multiple spaces)
                parts = line.split()
                label = int(parts[-1])
                text = " ".join(parts[:-1])

            texts.append(text)
            labels.append(label)
    return texts, labels


def get_custom_classification_dataloader(file_paths: list, 
                                         dataset_type: str = 'sentiment', 
                                         batch_size: int = 32, 
                                         max_seq_len: int = 64,
                                         validation_split: float = 0.2):
    """
    Generic loader for the 5 datasets.
    
    Args:
        file_paths: List of file paths to load (can be just one file).
        dataset_type: 'sentiment' (for IMDb/Amazon/Yelp) or 'mc_rp' (for MC/RP QNLP datasets).
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        validation_split: Fraction of data to use for validation.
    """
    all_texts = []
    all_labels = []

    # 1. Load Data
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        if dataset_type == 'mc_rp':
            t, l = load_mc_rp_data(path)
        elif dataset_type == 'sentiment':
            t, l = load_sentiment_data(path)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'sentiment' or 'mc_rp'.")
        
        all_texts.extend(t)
        all_labels.extend(l)

    print(f"Loaded {len(all_texts)} samples from {file_paths}")

    # 2. Tokenize
    # We train a fresh tokenizer on this dataset
    tokenizer = train_tokenizer(all_texts, vocab_size=5000) # Smaller vocab for small datasets
    print(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    # 3. Create Splits
    # Shuffle first
    indices = torch.randperm(len(all_texts)).tolist()
    split_val = int(len(all_texts) * validation_split)
    
    train_indices = indices[split_val:]
    val_indices = indices[:split_val]

    train_texts = [all_texts[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_texts = [all_texts[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    # 4. Create Datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_seq_len)

    def collate_fn(batch):
        inputs = jnp.array([item[0] for item in batch])
        targets = jnp.array([item[1] for item in batch])
        return inputs, targets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, tokenizer