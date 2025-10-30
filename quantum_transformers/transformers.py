from typing import Literal, Callable, Optional
import flax.linen as nn
import jax.numpy as jnp

from quantum_transformers.quantum_layer import QuantumLayer

# See:
# - https://nlp.sea        # Layer normalization
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py


class MultiHeadSelfAttention(nn.Module):
    hidden_size: int # Is the embedding size / input dimension, representing one token(one word or one patch) in a vector form
    num_heads: int # Number of attention heads
    dropout: float = 0.0 # Dropout rate

    quantum_w_shape: tuple = (1,) # Shape of the quantum layer weights
    quantum_circuit: Optional[Callable] = None # Quantum circuit function for the quantum layer

    @nn.compact
    def __call__(self, x, deterministic):
        batch_size, seq_len, hidden_size = x.shape
        # x.shape = (batch_size, seq_len, hidden_size)
        # x is the input sequence in tensor form
        assert hidden_size == self.hidden_size, f"Input hidden size ({hidden_size}) does not match layer hidden size ({self.hidden_size})" # Check input dimension
        assert hidden_size % self.num_heads == 0, f"Hidden size ({hidden_size}) must be divisible by the number of heads ({self.num_heads})" # Check divisibility
        head_dim = hidden_size // self.num_heads # Dimension of each attention head. It is the size of the subspace that each head operates on. The hidden size must be divisible by the number of heads to ensure that each head has an equal share of the total hidden size. The hidden size is split evenly among the heads.

        # Linear projections for queries, keys, and values
        if self.quantum_circuit is None:
            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
                for proj, x in zip([nn.Dense(features=hidden_size),
                                    nn.Dense(features=hidden_size),
                                    nn.Dense(features=hidden_size)], [x, x, x])
            ] # q, k, v shape = (batch_size, num_heads, seq_len, head_dim)
            # This works by applying each Dense layer to the input x to produce the queries, keys, and values.
            # zip is used to iterate over the three Dense layers and the input x simultaneously.
            # swapaxes is used to swap the second and third axes of the reshaped output, resulting in the desired shape of (batch_size, num_heads, seq_len, head_dim).
            # Each of the three projections (queries, keys, values) is computed using a separate Dense layer. The input x is passed through each Dense layer to produce q, k, and v. The output of each Dense layer has shape (batch_size, seq_len, hidden_size). This output is then reshaped to (batch_size, seq_len, num_heads, head_dim) and the axes are swapped to get the final shape of (batch_size, num_heads, seq_len, head_dim).
            
        # If a quantum circuit is provided, use QuantumLayer for projections
        else:
            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
                for proj, x in zip([QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit),
                                    QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit),
                                    QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)], [x, x, x])
            ]
            # hidden_size being the same as num_qubits means that the quantum layer is designed to handle input vectors of size hidden_size. Each token in the input sequence is represented as a vector of size hidden_size, and the quantum layer processes these vectors using a quantum circuit with hidden_size qubits. This allows the quantum layer to effectively capture complex relationships and patterns in the input data, leveraging the capabilities of quantum computing.
            # q, k, v shape = (batch_size, num_heads, seq_len, head_dim)
            # Note: In practice, it's more efficient to compute q, k, v in a single linear projection and then split the result. In quantum case, we keep them separate for clarity.
            

        # Compute scaled dot-product attention
        attn_logits = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(head_dim) 
        # @ is the matrix multiplication operator in Python. 
        # Here, q has shape (batch_size, num_heads, seq_len, head_dim) and k.swapaxes(-2, -1) has shape (batch_size, num_heads, head_dim, seq_len). The result of this matrix multiplication is attn_logits with shape (batch_size, num_heads, seq_len, seq_len).
        
        # attn_logits.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = nn.softmax(attn_logits, axis=-1)
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)

        # Compute output
        values = attn @ v # @ is the matrix multiplication operator in Python.
        # values.shape = (batch_size, num_heads, seq_len, head_dim)
        values = values.swapaxes(1, 2).reshape(batch_size, seq_len, hidden_size) 
        # values.shape = (batch_size, seq_len, hidden_size)
        if self.quantum_circuit is None:
            x = nn.Dense(features=hidden_size)(values)
        else:
            x = QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)(values)
        # x.shape = (batch_size, seq_len, hidden_size)

        return x


class FeedForward(nn.Module):
    hidden_size: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,) # Shape of the quantum layer weights
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(features=self.mlp_hidden_size)(x) # x.shape = (batch_size, seq_len, mlp_hidden_size)
        if self.quantum_circuit is not None:
            x = QuantumLayer(num_qubits=self.mlp_hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)(x) # x.shape = (batch_size, seq_len, mlp_hidden_size). The quantum layer processes the input x using a quantum circuit with mlp_hidden_size qubits. This allows the quantum layer to effectively capture complex relationships and patterns in the input data, leveraging the capabilities of quantum computing.
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic) #  This part applies to both classical and quantum cases. The nn.Dropout layer randomly sets a fraction of the input units to zero during training, which helps prevent overfitting and improves the model's generalization. The deterministic parameter ensures that dropout is only applied during training and not during evaluation or inference.
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        return x


class TransformerBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        attn_output = nn.LayerNorm()(x)
        attn_output = MultiHeadSelfAttention(hidden_size=self.hidden_size, num_heads=self.num_heads, dropout=self.dropout,
                                             quantum_circuit=self.quantum_attn_circuit)(attn_output, deterministic=deterministic)
        attn_output = nn.Dropout(rate=self.dropout)(attn_output, deterministic=deterministic)
        x = x + attn_output

        y = nn.LayerNorm()(x)
        y = FeedForward(hidden_size=self.hidden_size, mlp_hidden_size=self.mlp_hidden_size,
                        quantum_circuit=self.quantum_mlp_circuit)(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)

        return x + y


class Transformer(nn.Module):
    num_tokens: int
    max_seq_len: int
    hidden_size: int
    num_heads: int
    num_transformer_blocks: int
    mlp_hidden_size: int
    task: Literal['classification', 'mlm'] = 'classification'
    num_classes: int = 2  # Default for binary classification
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, train):
        # Token embedding
        x = nn.Embed(num_embeddings=self.num_tokens, features=self.hidden_size)(x)
        
        # Positional embedding
        x += nn.Embed(num_embeddings=self.max_seq_len, features=self.hidden_size)(jnp.arange(x.shape[1]))
        
        # Dropout
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, deterministic=not train)

        # Layer normalization
        x = nn.LayerNorm()(x)

        if self.task == 'classification':
            # Global average pooling for classification
            # Global average pooling means averaging the features across the sequence length dimension. This results in a fixed-size representation of the input sequence, regardless of its original length. This pooled representation can then be used for classification tasks.
            x = jnp.mean(x, axis=1)  # x.shape = (batch_size, hidden_size)
            x = nn.Dense(self.num_classes)(x)  # x.shape = (batch_size, num_classes)
            
            
        else:  # MLM task
            # For MLM, we need to predict the original token for each position
            x = nn.Dense(self.num_tokens)(x)  # x.shape = (batch_size, seq_len, num_tokens)
            # Each position will have logits for the entire vocabulary
            # What differsentiates MLM from other tasks is that the model is trained to predict the original tokens for only the masked positions in the input sequence, while ignoring the unmasked positions. This is typically achieved by setting the labels for unmasked positions to a special value (e.g., -100) that indicates they should be ignored during loss computation. This way, the model focuses on learning to predict the masked tokens based on their surrounding context.
            # What differs this part with the classification task is that in classification, we condense the sequence information into a single vector (using global average pooling) and then classify that vector. In contrast, for MLM, we maintain the sequence structure and predict a token for each position in the sequence.
        
        return x