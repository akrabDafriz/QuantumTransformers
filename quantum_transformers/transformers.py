from typing import Literal, Callable, Optional
import flax.linen as nn
import jax.numpy as jnp

from quantum_transformers.quantum_layer import QuantumLayer

# See:
# - https://nlp.seas.harvard.edu/2018/04/03/attention.html
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py


class MultiHeadSelfAttention(nn.Module):
    hidden_size: int 
    num_heads: int 
    dropout: float = 0.0 

    quantum_w_shape: tuple = (1,) # Shape of the quantum layer weights
    quantum_circuit: Optional[Callable] = None 

    @nn.compact
    def __call__(self, x, deterministic):
        batch_size, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size, f"Input hidden size ({hidden_size}) does not match layer hidden size ({self.hidden_size})" 
        assert hidden_size % self.num_heads == 0, f"Hidden size ({hidden_size}) must be divisible by the number of heads ({self.num_heads})" 
        head_dim = hidden_size // self.num_heads 

        # Linear projections for queries, keys, and values
        if self.quantum_circuit is None:
            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
                for proj, x in zip([nn.Dense(features=hidden_size),
                                    nn.Dense(features=hidden_size),
                                    nn.Dense(features=hidden_size)], [x, x, x])
            ] 
            
        # If a quantum circuit is provided, use QuantumLayer for projections
        else:
            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
                for proj, x in zip([QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit),
                                    QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit),
                                    QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)], [x, x, x])
            ]

        # Compute scaled dot-product attention
        attn_logits = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(head_dim) 
        
        attn = nn.softmax(attn_logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)

        # Compute output
        values = attn @ v 
        values = values.swapaxes(1, 2).reshape(batch_size, seq_len, hidden_size) 
        
        if self.quantum_circuit is None:
            x = nn.Dense(features=hidden_size)(values)
        else:
            x = QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)(values)

        return x


class FeedForward(nn.Module):
    hidden_size: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,) 
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        # 1. First Dense Layer (or Quantum Layer)
        if self.quantum_circuit is None:
            x = nn.Dense(features=self.mlp_hidden_size)(x)
        else:
            x = QuantumLayer(num_qubits=self.mlp_hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)(x)
            
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.gelu(x)
        
        # 2. Second Dense Layer (Usually keeps classical for dimensionality restoration, 
        # but could be quantum if dims match. Following original pattern of strictly projecting up then down)
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
        
        # FIX: Pass quantum_w_shape to MultiHeadSelfAttention
        attn_output = MultiHeadSelfAttention(
            hidden_size=self.hidden_size, 
            num_heads=self.num_heads, 
            dropout=self.dropout,
            quantum_w_shape=self.quantum_w_shape,  # <--- FIX
            quantum_circuit=self.quantum_attn_circuit
        )(attn_output, deterministic=deterministic)
        
        attn_output = nn.Dropout(rate=self.dropout)(attn_output, deterministic=deterministic)
        x = x + attn_output

        y = nn.LayerNorm()(x)
        
        # FIX: Pass quantum_w_shape to FeedForward
        y = FeedForward(
            hidden_size=self.hidden_size, 
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_w_shape=self.quantum_w_shape,  # <--- FIX
            quantum_circuit=self.quantum_mlp_circuit
        )(y, deterministic=deterministic)
        
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
    num_classes: int = 2 
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
                quantum_w_shape=self.quantum_w_shape,      # <--- FIX: Propagate down
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, deterministic=not train)

        # Layer normalization
        x = nn.LayerNorm()(x)

        if self.task == 'classification':
            x = jnp.mean(x, axis=1)  # Global Average Pooling
            x = nn.Dense(self.num_classes)(x)  
        else:  # MLM task
            x = nn.Dense(self.num_tokens)(x)  
        
        return x