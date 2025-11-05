# --- 1. FRAMEWORK SETUP (MUST BE FIRST) ---
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
# This MUST run before JAX is imported.
tf.config.set_visible_devices([], device_type='GPU')

import jax
# --- END FRAMEWORK SETUP ---

import os
import pickle
import jax.numpy as jnp
from flax import serialization

# --- NEW IMPORTS for creating our own tokenizer ---
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
# --- MODIFICATION: Import PreTrainedTokenizerFast ---
from transformers import PreTrainedTokenizerFast
# --- END NEW IMPORTS ---

# Import custom modules
# --- MODIFICATION: Restore original import ---
from quantum_transformers.datasets import get_mlm_dataloaders
from quantum_transformers.training import train_and_evaluate
from quantum_transformers.transformers import Transformer
from quantum_transformers.quantum_layer import get_circuit
# We only need 'save_model' from inference.py for this script
from quantum_transformers.inference import save_model

# --- 2. PATHS AND DIRECTORIES ---
print("Setting up environment...")

# Define directories
data_dir = './data' # <-- This directory will now be used for caching
CLASSICAL_MODEL_PATH = './models/mlm_classical'
QUANTUM_MODEL_PATH = './models/mlm_quantum'
os.makedirs(CLASSICAL_MODEL_PATH, exist_ok=True)
os.makedirs(QUANTUM_MODEL_PATH, exist_ok=True)
os.makedirs(data_dir, exist_ok=True) # <-- Ensure the data directory exists

# Print JAX devices
print("Available JAX devices:")
for d in jax.devices():
    print(f"- {d} ({d.device_kind})")

# --- 3. PREPARE SMALL TOKENIZER ---
print("\nPreparing small tokenizer...")

VOCAB_SIZE = 1000  # Our new, small vocabulary size
# --- MODIFICATION: This is now a DIRECTORY path ---
TOKENIZER_DIR = os.path.join(CLASSICAL_MODEL_PATH, 'tinystories_tokenizer_directory')

# --- MODIFICATION: Check for the directory ---
if not os.path.exists(TOKENIZER_DIR):
    print(f"Tokenizer not found. Training a new one with vocab size {VOCAB_SIZE}...")
    
    print("Loading 50k examples for tokenizer training (to save RAM)...")
    dataset = load_dataset(
        "roneneldan/TinyStories", 
        split="train[:50000]", # <-- SLICE THE DATASET
        cache_dir=data_dir
    )

    # Instantiate a new tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # Create a generator to feed text to the trainer efficiently
    def get_training_corpus():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # --- MODIFICATION: Wrap and save using transformers ---
    print("Wrapping tokenizer with PreTrainedTokenizerFast...")
    # Wrap the trained 'tokenizers' object into a 'transformers' compatible object
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    
    # Save it as a pretrained tokenizer (creates a directory with config files)
    hf_tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"New tokenizer saved to {TOKENIZER_DIR}")
else:
    print(f"Found existing tokenizer at {TOKENIZER_DIR}")


# --- 4. REMOVED SELF-CONTAINED DATALOADER FUNCTION ---
# This logic is now in quantum_transformers/datasets.py


# --- 5. LOAD DATA ---
print("\nLoading and preparing dataset...")

# Set data loading parameters
block_size = 128  # The size of our text chunks
batch_size = 8   # How many chunks to process at once

# Get the dataloaders and the tokenizer
# --- MODIFICATION: Call get_mlm_dataloaders from the imported module ---
(train_dataloader_gen, val_dataloader_gen, test_dataloader_gen), tokenizer = get_mlm_dataloaders(
    dataset_name='roneneldan/TinyStories',
    model_checkpoint=TOKENIZER_DIR,
    block_size=block_size,
    batch_size=batch_size,
    data_dir=data_dir
)

print(f"\nDataset loading complete.")
print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

# Get one batch for model initialization
print("\nGetting one batch to verify dataloader...")
try:
    init_batch_tuple = next(iter(train_dataloader_gen()))
    init_batch_input = init_batch_tuple[0]
    print(f"Initialization batch shape: {init_batch_input.shape}")
except StopIteration:
    print("ERROR: Training dataloader is empty. Cannot initialize models.")
    exit()


# --- 6. TRAIN CLASSICAL MODEL ---
print("\n--- 6. Starting Classical Transformer Training ---")

classical_model = Transformer(
    num_tokens=len(tokenizer.vocab),  # This will now be 1000
    max_seq_len=block_size,
    task='mlm',
    hidden_size=8,
    num_heads=2,
    num_transformer_blocks=4,
    mlp_hidden_size=8,
    dropout=0.1
)

(classical_test_loss, classical_test_ppl), classical_best_state = train_and_evaluate(
    model=classical_model,
    train_dataloader=train_dataloader_gen,
    val_dataloader=val_dataloader_gen,
    test_dataloader=test_dataloader_gen,
    task='mlm',
    num_epochs=40
)

print("\n--- Classical Transformer Training Finished ---")
print(f"Final Test Perplexity: {classical_test_ppl:.4f}")

# Save the classical model
save_model(classical_best_state, tokenizer, CLASSICAL_MODEL_PATH)

# --- 7. TRAIN QUANTUM MODEL ---
print("\n--- 7. Starting Quantum Transformer Training ---")

quantum_model = Transformer(
    num_tokens=len(tokenizer.vocab), # This will now be 1000
    max_seq_len=block_size,
    task='mlm',
    hidden_size=8,
    num_heads=2,
    num_transformer_blocks=4,
    mlp_hidden_size=8,
    dropout=0.1,
    quantum_attn_circuit=get_circuit(),  # Activate the quantum attention
    quantum_mlp_circuit=get_circuit()    # Activate the quantum MLP
)

(quantum_test_loss, quantum_test_ppl), quantum_best_state = train_and_evaluate(
    model=quantum_model,
    train_dataloader=train_dataloader_gen,
    val_dataloader=val_dataloader_gen,
    test_dataloader=test_dataloader_gen,
    task='mlm',
    num_epochs=40
)

print("\n--- Quantum Transformer Training Finished ---")
print(f"Final Test Perplexity: {quantum_test_ppl:.4f}")

# Save the quantum model
save_model(quantum_best_state, tokenizer, QUANTUM_MODEL_PATH)

print("\n--- Classical Transformer Training Finished Earlier ---")
print(f"Final Test Perplexity: {classical_test_ppl:.4f}")

print("\n--- Training and Saving Complete ---")