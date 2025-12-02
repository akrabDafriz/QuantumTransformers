import os
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# --- 1. SCRIPT PARAMETERS ---
def get_args():
    parser = argparse.ArgumentParser(description="Train and check a tokenizer for [UNK] tokens.")
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="The target vocabulary size to train."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to cache datasets."
    )
    parser.add_argument(
        "--num_examples_train",
        type=int,
        default=100000,
        help="Number of examples to use for training the tokenizer."
    )
    parser.add_argument(
        "--num_examples_check",
        type=int,
        default=220000,
        help="Number of examples to check for [UNK] tokens (matches your train+val+test splits)."
    )
    return parser.parse_args()

# --- 2. DATASET ITERATOR (from mlm_training_2.py) ---
def get_text_iterator(data_dir, num_examples):
    """
    Returns an iterator over the text content of the TinyStories dataset
    for tokenizer training.
    """
    print(f"Loading 'roneneldan/TinyStories' split='train[:{num_examples}]' for tokenizer training...")
    dataset = load_dataset('roneneldan/TinyStories', split=f'train[:{num_examples}]', cache_dir=data_dir)
    print("Dataset loaded. Yielding text...")
    for i in range(len(dataset)):
        yield dataset[i]['text']

# --- 3. MAIN FUNCTION ---
def main():
    args = get_args()
    print(f"--- Starting Tokenizer Check ---")
    print(f"Target Vocab Size: {args.vocab_size}")
    os.makedirs(args.data_dir, exist_ok=True)
    
    # --- 4. TRAIN TOKENIZER (from mlm_training_2.py) ---
    print(f"\nTraining tokenizer with vocab_size={args.vocab_size}...")
    
    # Initialize a base tokenizer object
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # This is the FIX (Bug 1) from your script
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Train the tokenizer
    tokenizer.train_from_iterator(
        get_text_iterator(args.data_dir, args.num_examples_train), 
        trainer=trainer
    )
    print("Tokenizer training complete.")
    
    # Wrap in PreTrainedTokenizerFast to make it compatible with .map()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=512, # Use a generic max length for this check
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    
    unk_token_id = tokenizer.unk_token_id
    print(f"\nTokenizer wrapped. Actual vocab size: {tokenizer.vocab_size}")
    print(f"ID for [UNK] token is: {unk_token_id}")

    # --- 5. CHECK FOR [UNK] TOKENS ---
    print(f"\nLoading 'roneneldan/TinyStories' split='train[:{args.num_examples_check}]' to check for [UNK]s...")
    dataset = load_dataset('roneneldan/TinyStories', split=f'train[:{args.num_examples_check}]', cache_dir=args.data_dir)
    
    def tokenize_function(examples):
        # Tokenize without padding or truncation to check all tokens
        return tokenizer(examples["text"], padding=False, truncation=False)

    print("Tokenizing the entire dataset (this may take a moment)...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, # Use multiple processes to speed up
        remove_columns=["text"],
        load_from_cache_file=False  # <--- ADD THIS LINE
    )
    print("Tokenization complete.")
    
    print("\nCounting [UNK] tokens...")
    total_tokens = 0
    unk_tokens = 0
    
    # Iterate through all tokenized examples to count
    for input_ids in tqdm(tokenized_dataset['input_ids'], desc="Counting"):
        total_tokens += len(input_ids)
        for token_id in input_ids:
            if token_id == unk_token_id:
                unk_tokens += 1
                
    # --- 6. REPORT ---
    print("\n--- Results ---")
    print(f"Target Vocab Size:    {args.vocab_size}")
    print(f"Actual Vocab Size:    {tokenizer.vocab_size}")
    print(f"Total Tokens Checked: {total_tokens:,}")
    print(f"Total [UNK] Tokens:   {unk_tokens:,}")
    
    if unk_tokens == 0:
        print("\n✅ SUCCESS: No [UNK] tokens found!")
    else:
        print(f"\n⚠️ WARNING: Found {unk_tokens} [UNK] tokens.")
    print("-----------------")


if __name__ == "__main__":
    main()