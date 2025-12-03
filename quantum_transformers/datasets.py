import os
import jax.numpy as jnp
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset
import torch
import math

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
    if not file_path or not os.path.exists(file_path):
        return [], []

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
    if not file_path or not os.path.exists(file_path):
        return [], []

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
                parts = line.split()
                label = int(parts[-1])
                text = " ".join(parts[:-1])

            texts.append(text)
            labels.append(label)
    return texts, labels


def get_custom_classification_dataloader(
    dataset_type: str,
    file_paths: list = None,
    train_path: str = None,
    val_path: str = None,
    test_path: str = None,
    batch_size: int = 32, 
    max_seq_len: int = 64
):
    """
    Generic loader for datasets.
    
    Args:
        dataset_type: 'sentiment' (Auto-split 80/10/10) or 'mc_rp' (Uses explicit train/val/test paths).
        file_paths: List of file paths (Used ONLY for 'sentiment' mode).
        train_path: Path to training file (Used ONLY for 'mc_rp' mode).
        val_path: Path to validation/dev file (Used ONLY for 'mc_rp' mode).
        test_path: Path to test file (Used ONLY for 'mc_rp' mode).
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    
    # --- 1. Load Data Based on Strategy ---
    
    if dataset_type == 'sentiment':
        # Strategy: Load all files, merge, then split 80/10/10
        if not file_paths:
            raise ValueError("For 'sentiment' type, 'file_paths' list must be provided.")
            
        print(f"Loading sentiment data from: {file_paths}")
        all_texts = []
        all_labels = []
        for path in file_paths:
            t, l = load_sentiment_data(path)
            all_texts.extend(t)
            all_labels.extend(l)
            
        # Shuffle
        total_count = len(all_texts)
        indices = torch.randperm(total_count).tolist()
        
        # Calculate split sizes (800/100/100 for a 1000 dataset)
        train_end = int(total_count * 0.8)
        val_end = int(total_count * 0.9)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_texts = [all_texts[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        
        val_texts = [all_texts[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        
        test_texts = [all_texts[i] for i in test_indices]
        test_labels = [all_labels[i] for i in test_indices]
        
        # Tokenizer training data: Use Train set (standard practice) or All (if data is very small)
        # For these datasets, we'll use all data to ensure coverage
        tokenizer_corpus = all_texts

    elif dataset_type == 'mc_rp':
        # Strategy: Use explicit splits. No shuffling between files.
        if not train_path:
            raise ValueError("For 'mc_rp' type, 'train_path' must be provided.")
            
        print(f"Loading MC/RP data.")
        print(f"Train: {train_path}")
        print(f"Val:   {val_path}")
        print(f"Test:  {test_path}")

        train_texts, train_labels = load_mc_rp_data(train_path)
        
        if val_path:
            val_texts, val_labels = load_mc_rp_data(val_path)
        else:
            val_texts, val_labels = [], [] # Handle RP case (no validation set)
            
        if test_path:
            test_texts, test_labels = load_mc_rp_data(test_path)
        else:
            test_texts, test_labels = [], []

        # Aggregate for tokenizer
        tokenizer_corpus = train_texts + val_texts + test_texts

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'sentiment' or 'mc_rp'.")

    print(f"Data Loaded. Counts -> Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # --- 2. Tokenize ---
    tokenizer = train_tokenizer(tokenizer_corpus, vocab_size=5000)
    print(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    # --- 3. Create Datasets ---
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
    
    # Handle empty validation/test sets gracefully
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_seq_len) if val_texts else None
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len) if test_texts else None

    # --- 4. Create Loaders ---
    def collate_fn(batch):
        inputs = jnp.array([item[0] for item in batch])
        targets = jnp.array([item[1] for item in batch])
        return inputs, targets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if test_dataset else None

    return train_loader, val_loader, test_loader, tokenizer