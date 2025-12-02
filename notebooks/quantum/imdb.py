import jax

from quantum_transformers.datasets import get_imdb_dataloaders
from quantum_transformers.training import train_and_evaluate
from quantum_transformers.transformers import Transformer
from quantum_transformers.quantum_layer import get_circuit

# Data directory from the notebook
data_dir = './imdb-data'

print("Checking JAX devices:")
for d in jax.devices():
    print(d, d.device_kind)

# --- Parameters and Data Loading ---

# Define the parameters
batch_size = 32
max_vocab_size = 20_000
max_seq_len = 512

# Load the data
print("Loading IMDb dataset...")
(train_dataloader, val_dataloader, test_dataloader), vocab, tokenizer = get_imdb_dataloaders(
    data_dir='./data', 
    batch_size=batch_size, 
    max_vocab_size=max_vocab_size, 
    max_seq_len=max_seq_len
)

# Print the results
print(f"Vocabulary size: {len(vocab)}")
print("First batch (tokenized):")
first_batch = next(iter(train_dataloader))
print(first_batch[0][0])
print("\nFirst batch (detokenized):")
print(' '.join(map(bytes.decode, tokenizer.detokenize(first_batch[0])[0].numpy().tolist())))


# --- Model Training and Evaluation ---

print("\nInitializing model...")
model = Transformer(num_tokens=len(vocab), max_seq_len=512, num_classes=2, hidden_size=6, num_heads=2, num_transformer_blocks=4, mlp_hidden_size=3,
                    quantum_attn_circuit=get_circuit(), quantum_mlp_circuit=get_circuit())

print("Starting training and evaluation...")
# change num_epochs as needed
train_and_evaluate(model, train_dataloader, val_dataloader, test_dataloader, num_classes=2, num_epochs=1)

print("Script finished.")