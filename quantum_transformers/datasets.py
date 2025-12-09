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


def train_tokenizer(texts, tokenizer_type='bpe', vocab_size=20000):
    """
    Trains a tokenizer (BPE or WordLevel).
    """
    if tokenizer_type == 'wordlevel':
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # WordLevelTrainer doesn't strictly enforce vocab_size in the same way, 
        # but we can set min_frequency if needed. For now, we just map all words.
        trainer = trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"])
    else:
        # BPE
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]"])
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def load_mc_rp_data(file_path):
    """Parses MC/RP datasets."""
    if not file_path or not os.path.exists(file_path):
        return [], []

    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label = int(parts[0])
            text = " ".join(parts[1:])
            labels.append(label)
            texts.append(text)
    return texts, labels


def load_sentiment_data(file_path):
    """Parses IMDb/Amazon/Yelp datasets."""
    if not file_path or not os.path.exists(file_path):
        return [], []

    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                parts = line.split('\t')
                text = parts[0].strip()
                label = int(parts[-1])
            else:
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
    max_seq_len: int = 64,
    tokenizer_type: str = 'bpe',
    vocab_size: int = 20000,
    tokenizer_files: list = None
):
    """
    Args:
        tokenizer_type: 'bpe' or 'wordlevel'.
        vocab_size: Target vocabulary size (mostly for BPE).
        tokenizer_files: Optional list of files to train the tokenizer on. 
                         If None, uses the training data itself.
    """
    
    # --- 1. Load Data Based on Strategy ---
    if dataset_type == 'sentiment':
        if not file_paths:
            raise ValueError("For 'sentiment' type, 'file_paths' list must be provided.")
        
        # Load dataset for TRAINING/EVAL
        print(f"Loading sentiment data from: {file_paths}")
        all_texts, all_labels = [], []
        for path in file_paths:
            t, l = load_sentiment_data(path)
            all_texts.extend(t)
            all_labels.extend(l)
            
        # Shuffle & Split 80/10/10
        total_count = len(all_texts)
        indices = torch.randperm(total_count).tolist()
        
        train_end = int(total_count * 0.8)
        val_end = int(total_count * 0.9)
        
        train_texts = [all_texts[i] for i in indices[:train_end]]
        train_labels = [all_labels[i] for i in indices[:train_end]]
        
        val_texts = [all_texts[i] for i in indices[train_end:val_end]]
        val_labels = [all_labels[i] for i in indices[train_end:val_end]]
        
        test_texts = [all_texts[i] for i in indices[val_end:]]
        test_labels = [all_labels[i] for i in indices[val_end:]]

    elif dataset_type == 'mc_rp':
        # Load explicit splits
        train_texts, train_labels = load_mc_rp_data(train_path)
        val_texts, val_labels = load_mc_rp_data(val_path) if val_path else ([], [])
        test_texts, test_labels = load_mc_rp_data(test_path) if test_path else ([], [])

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    print(f"Data Loaded. Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # --- 2. Tokenize ---
    # Prepare corpus for Tokenizer Training
    if tokenizer_files:
        # If user explicitly provided files for the tokenizer (e.g. for Shared Sentiment Tokenizer)
        print(f"Training tokenizer on shared files: {tokenizer_files}")
        tokenizer_corpus = []
        for path in tokenizer_files:
            # We need to detect type to load correctly for tokenizer corpus
            if 'mc' in path or 'rp' in path:
                t, _ = load_mc_rp_data(path)
            else:
                t, _ = load_sentiment_data(path)
            tokenizer_corpus.extend(t)
    else:
        # Default: Train on the current training set (and val/test if available)
        tokenizer_corpus = train_texts + val_texts + test_texts

    tokenizer = train_tokenizer(tokenizer_corpus, tokenizer_type=tokenizer_type, vocab_size=vocab_size)
    print(f"Tokenizer ({tokenizer_type}) trained. Vocab size: {tokenizer.get_vocab_size()}")

    # --- 3. Create Datasets & Loaders ---
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_seq_len) if val_texts else None
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len) if test_texts else None

    def collate_fn(batch):
        inputs = jnp.array([item[0] for item in batch])
        targets = jnp.array([item[1] for item in batch])
        return inputs, targets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if test_dataset else None

    return train_loader, val_loader, test_loader, tokenizer