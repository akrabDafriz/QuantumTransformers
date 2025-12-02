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
from tokenizers import Tokenizer, pre_tokenizers # <--- Import pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
# --- MODIFICATION: Import PreTrainedTokenizerFast ---
from transformers import PreTrainedTokenizerFast
# --- END NEW IMPORTS ---

# Import custom modules
# --- MODIFICATION: Restore original import ---\
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
os.makedirs(data_dir, exist_ok=True)

# --- NEW: Define a dedicated path for the new tokenizer ---
TOKENIZER_PATH = os.path.join(CLASSICAL_MODEL_PATH, 'tinystories_tokenizer_directory')
os.makedirs(TOKENIZER_PATH, exist_ok=True)


# Print JAX devices
print("Available JAX devices:")
for d in jax.devices():
    print(f"- {d} ({d.device_kind})")

# --- 3. MODEL & DATA PARAMETERS ---
block_size = 128  # The size of our text chunks
batch_size = 16   # How many chunks to process at once
VOCAB_SIZE = 198 # 198 is the limit for TinyStories Dataset

# --- 4. PREPARE SMALL TOKENIZER ---
print("\nPreparing small tokenizer...")

def get_text_iterator(num_examples=100000):
    """
    Returns an iterator over the text content of the TinyStories dataset
    for tokenizer training. We use a slice to speed up training.
    """
    print(f"Loading 'roneneldan/TinyStories' split='train[:{num_examples}]' for tokenizer training...")
    dataset = load_dataset('roneneldan/TinyStories', split=f'train[:{num_examples}]', cache_dir=data_dir)
    print("Dataset loaded. Yielding text...")
    for i in range(len(dataset)):
        yield dataset[i]['text']

# Initialize a base tokenizer object
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# --- THIS IS THE FIX (Bug 1) ---
# We MUST define the pre_tokenizer on the object, otherwise
# it doesn't know how to split sentences into words.
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# --- END FIX ---

tokenizer_file = os.path.join(TOKENIZER_PATH, "tokenizer.json")

if not os.path.exists(tokenizer_file):
    print(f"No tokenizer found at {tokenizer_file}. Training new tokenizer...")
    # tokenizer.pre_tokenizer is already set
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    # Train the tokenizer
    tokenizer.train_from_iterator(get_text_iterator(), trainer=trainer)
    # Save the tokenizer
    tokenizer.save(tokenizer_file)
    print(f"New tokenizer trained and saved to {tokenizer_file}")
else:
    print(f"Found existing tokenizer. Loading from {tokenizer_file}...")
    
    # Load the trained tokenizer from file
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    # --- ADDED SAFETY (Bug 1) ---
    # The loaded JSON *should* contain the pre-tokenizer, but we
    # set it again just to be 100% sure the loaded object has it.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # --- END SAFETY ADD ---


print("\nTokenizer preparation complete. Wrapping in PreTrainedTokenizerFast...")

# This wrapper makes it compatible with the Hugging Face datasets library
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=block_size,
    pad_token="[PAD]",
    unk_token="[UNK]",
    mask_token="[MASK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)

print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
if tokenizer.vocab_size != VOCAB_SIZE:
     print(f"Warning: Loaded tokenizer vocab size ({tokenizer.vocab_size}) does not match target ({VOCAB_SIZE}).")


# --- 5. LOAD DATA ---
print("\nLoading and preparing dataset...")

# Get the dataloaders and the tokenizer
(train_dataloader_gen, val_dataloader_gen, test_dataloader_gen), tokenizer_from_data = get_mlm_dataloaders(
    dataset_name='roneneldan/TinyStories',
    # model_checkpoint=None, # We are not using a pretrained one
    block_size=block_size,
    batch_size=batch_size,
    data_dir=data_dir,
    tokenizer=tokenizer # <-- Pass our new, trained tokenizer
)

print(f"\nDataset loading complete.")
# Use the tokenizer we passed, as it's the definitive one
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

# Get one batch for model initialization
print("\nGetting one batch to verify dataloader...")
try:
    init_batch_tuple = next(iter(train_dataloader_gen()))
    init_batch_input = init_batch_tuple[0]
    print(f"Initialization batch shape: {init_batch_input.shape}")
    print(f"Example input_ids (first 10): {init_batch_input[0, :10]}")
except StopIteration:
    print("ERROR: Training dataloader is empty. Cannot initialize models.")
    exit()


# --- 6. TRAIN CLASSICAL MODEL ---
print("\n--- 6. Starting Classical Transformer Training ---")

classical_model = Transformer(
    num_tokens=tokenizer.vocab_size, # This will now be 1000
    max_seq_len=block_size,
    task='mlm',
    hidden_size=8,
    num_heads=2,
    num_transformer_blocks=8,
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
# We pass the PreTrainedTokenizerFast object for saving
save_model(classical_best_state, tokenizer, CLASSICAL_MODEL_PATH)

# --- 7. TRAIN QUANTUM MODEL ---
print("\n--- 7. Starting Quantum Transformer Training ---")

quantum_model = Transformer(
    num_tokens=tokenizer.vocab_size, # This will now be 1000
    max_seq_len=block_size,
    task='mlm',
    hidden_size=8,
    num_heads=2,
    num_transformer_blocks=8,
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