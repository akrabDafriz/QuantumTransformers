import jax
import tensorflow as tf
import os
import pickle
import jax.numpy as jnp
from flax import serialization

# Import custom modules
from quantum_transformers.datasets import get_mlm_dataloaders
from quantum_transformers.training import train_and_evaluate
from quantum_transformers.transformers import Transformer
from quantum_transformers.quantum_layer import get_circuit
# We only need 'save_model' from inference.py for this script
from quantum_transformers.inference import save_model

# --- 1. SETUP ---
print("Setting up environment...")

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

# Define directories
data_dir = './data'
CLASSICAL_MODEL_PATH = './models/mlm_classical'
QUANTUM_MODEL_PATH = './models/mlm_quantum'
os.makedirs(CLASSICAL_MODEL_PATH, exist_ok=True)
os.makedirs(QUANTUM_MODEL_PATH, exist_ok=True)

# Print JAX devices
print("Available JAX devices:")
for d in jax.devices():
    print(f"- {d} ({d.device_kind})")

# --- 2. LOAD DATA ---
print("\nLoading and preparing dataset...")

# Set data loading parameters
block_size = 128  # The size of our text chunks
batch_size = 8   # How many chunks to process at once

# Get the dataloaders and the tokenizer
(train_dataloader_gen, val_dataloader_gen, test_dataloader_gen), tokenizer = get_mlm_dataloaders(
    dataset_name='Helsinki-NLP/opus_books',
    model_checkpoint='bert-base-uncased',
    block_size=block_size,
    batch_size=batch_size
)

print(f"\nDataset loading complete.")
print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

# Get one batch for model initialization
# We don't need this for training, but it's good to check that the dataloader isn't empty
print("\nGetting one batch to verify dataloader...")
try:
    init_batch_tuple = next(iter(train_dataloader_gen()))
    init_batch_input = init_batch_tuple[0]
    print(f"Initialization batch shape: {init_batch_input.shape}")
except StopIteration:
    print("ERROR: Training dataloader is empty. Cannot initialize models.")
    exit()


# --- 3. TRAIN CLASSICAL MODEL ---
print("\n--- 3. Starting Classical Transformer Training ---")

classical_model = Transformer(
    num_tokens=len(tokenizer.vocab),
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
    num_epochs=1  # Change this for different training epoch numbers
)

print("\n--- Classical Transformer Training Finished ---")
print(f"Final Test Perplexity: {classical_test_ppl:.4f}")

# Save the classical model
save_model(classical_best_state, tokenizer, CLASSICAL_MODEL_PATH)

# --- 4. TRAIN QUANTUM MODEL ---
print("\n--- 4. Starting Quantum Transformer Training ---")

quantum_model = Transformer(
    num_tokens=len(tokenizer.vocab),
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
    num_epochs=1  # Change this for different training epoch numbers
)

print("\n--- Quantum Transformer Training Finished ---")
print(f"Final Test Perplexity: {quantum_test_ppl:.4f}")

# Save the quantum model
save_model(quantum_best_state, tokenizer, QUANTUM_MODEL_PATH)

print("\n--- Classical Transformer Training Finished Earlier ---")
print(f"Final Test Perplexity: {classical_test_ppl:.4f}")

print("\n--- Training and Saving Complete ---")

