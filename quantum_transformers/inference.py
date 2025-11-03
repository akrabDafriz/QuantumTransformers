import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import train_state
import pickle
import os
from functools import partial

from .transformers import Transformer

def save_model(state: train_state.TrainState, tokenizer, save_path: str):
    """
    Saves the model parameters and the tokenizer to a directory.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save model parameters
    param_bytes = serialization.to_bytes(state.params)
    with open(os.path.join(save_path, 'model_params.msgpack'), 'wb') as f:
        f.write(param_bytes)
    
    # 2. Save the tokenizer
    with open(os.path.join(save_path, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print(f"Model and tokenizer saved to {save_path}")

def load_model(model_path: str, model_instance: Transformer, init_batch):
    """
    Loads the model parameters and the tokenizer from a directory.
    
    Args:
        model_path: The directory where the model and tokenizer are saved.
        model_instance: An *instance* of your Transformer model (to get the structure).
        init_batch: A single batch of data (e.g., from a dataloader) to initialize
                      a new model state with the correct shapes.

    Returns:
        (params, tokenizer): A tuple containing the loaded parameters and tokenizer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
        
    # 1. Load the tokenizer
    with open(os.path.join(model_path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
        
    # 2. Load model parameters
    with open(os.path.join(model_path, 'model_params.msgpack'), 'rb') as f:
        param_bytes = f.read()
        
    # Initialize a new model to get the correct pytree structure
    key = jax.random.PRNGKey(0)
    # Ensure init_batch doesn't have labels if it's a tuple
    if isinstance(init_batch, tuple):
        init_batch = init_batch[0]
        
    params_template = model_instance.init({'params': key, 'dropout': key}, init_batch, train=False)['params']
    
    # Load the saved parameters into the template structure
    loaded_params = serialization.from_bytes(params_template, param_bytes)
    
    print(f"Model and tokenizer loaded from {model_path}")
    return loaded_params, tokenizer

# This tells JAX to treat the first argument (model_apply_fn) as a static
# part of the function, not as an array to be traced.
@partial(jax.jit, static_argnums=0)
def _get_predictions(model_apply_fn, params, token_ids):
    """JIT-compiled function for model prediction."""
    return model_apply_fn({'params': params}, token_ids, train=False)

def predict_masked_token(text: str, model: Transformer, params, tokenizer, top_k=5):
    """
    Performs fill-in-the-blank inference on a string with one [MASK] token.
    
    Args:
        text: The input string, e.g., "The capital of France is [MASK]."
        model: The Transformer model instance.
        params: The loaded model parameters.
        tokenizer: The loaded tokenizer.
        top_k: The number of top predictions to return.
    """
    # 1. Tokenize the input text
    inputs = tokenizer(text, return_tensors='np', padding='max_length', 
                       truncation=True, max_length=model.max_seq_len)
    token_ids = inputs['input_ids']
    
    # 2. Find the position of the [MASK] token
    try:
        mask_token_id = tokenizer.mask_token_id
        mask_position = jnp.where(token_ids[0] == mask_token_id)[0][0] # This is the position of the [MASK] token
    except (AttributeError, IndexError):
        print(f"Error: Could not find [MASK] token (ID: {tokenizer.mask_token_id}) in input text.")
        print(f"Tokenized IDs: {token_ids[0]}")
        return

    # 3. Get model predictions (logits)
    logits = _get_predictions(model.apply, params, token_ids)
    
    # 4. Get the logits for the masked token's position
    mask_logits = logits[0, mask_position, :]
    
    # 5. Find the top_k most likely tokens
    top_k_indices = jnp.argsort(mask_logits)[-top_k:][::-1]
    top_k_logits = mask_logits[top_k_indices]
    
    # 6. Decode the tokens back to words
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
    
    print(f"Input: '{text}'")
    print("Top predictions:")
    for token, logit in zip(predicted_tokens, top_k_logits):
        print(f"  - {token:<15} (Logit: {logit:.2f})")

    return list(zip(predicted_tokens, top_k_logits))

def evaluate_on_list(texts: list, model: Transformer, params, tokenizer, top_k=1):
    """
    Performs fill-in-the-blank inference on a list of sentences.
    
    Args:
        texts: A list of strings, e.g., ["Sentence 1 [MASK].", "Sentence 2 [MASK]."]
        model: The Transformer model instance.
        params: The loaded model parameters.
        tokenizer: The loaded tokenizer.
        top_k: The number of top predictions to show for each sentence.
    """
    print(f"--- Running batch inference on {len(texts)} sentences ---")
    
    for i, text in enumerate(texts):
        print(f"\nExample {i+1}:")
        predict_masked_token(
            text=text,
            model=model,
            params=params,
            tokenizer=tokenizer,
            top_k=top_k
        )
    print("\n--- Batch inference complete ---")