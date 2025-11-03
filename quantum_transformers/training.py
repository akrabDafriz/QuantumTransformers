import jax
import jax.numpy as jnp
import optax
import time
from flax.training import train_state
from flax.training.common_utils import onehot
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Helper function for MLM cross-entropy loss
def cross_entropy_loss(logits, labels, num_classes):
    """
    Calculates cross-entropy loss for MLM, handling -100 labels without boolean indexing
    """
    logits_flat = logits.reshape(-1, num_classes)
    labels_flat = labels.reshape(-1)
    
    # The difference between cross entropy loss for classification and MLM is that in MLM, some labels are -100 (ignore index).
    # We need to ignore these positions in the loss calculation. The rest positions should contribute to the loss.
    # We can do this by creating a mask where labels are not -100, and then using this mask to weight the loss.
    
    # Instead of boolean indexing, use weights
    weights = jnp.where(labels_flat != -100, 1.0, 0.0)
    
    # Calculate loss for all positions
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_flat,
        labels=jnp.where(labels_flat != -100, labels_flat, 0)  # Replace -100 with 0 to avoid invalid labels
    )
    
    # Apply weights to mask out the -100 positions
    losses = losses * weights
    
    # Calculate mean over non-masked positions
    return jnp.sum(losses) / (jnp.sum(weights) + 1e-8)  # Add small epsilon to prevent division by zero

def train_and_evaluate(model, train_dataloader, val_dataloader, test_dataloader, num_epochs,
                       task='classification'):

    # Initialize model and optimizer
    key = jax.random.PRNGKey(0)
    key, dropout_key = jax.random.split(key)
    
    # Create a fresh generator and get the first batch for initialization
    # We need the dataloader to be callable to get a fresh iterator
    train_loader_iter = train_dataloader()
    init_batch = next(iter(train_loader_iter))[0]
    
    params = model.init({'params': key, 'dropout': dropout_key}, init_batch, train=False)['params']

    print(f"Number of parameters = {sum(p.size for p in jax.tree_util.tree_leaves(params))}")
    
    # For classification, num_classes comes from the function argument. For MLM, it's the vocab size.
    num_output_classes = model.num_tokens if task == 'mlm' else 2

    optimizer = optax.adam(learning_rate=3e-5)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    # JIT compile the train and eval steps for performance
    @jax.jit
    def train_step(state, batch, dropout_key):
        inputs, labels = batch
        
        def loss_fn(params):
            # Pass the input tensor directly to the model
            logits = state.apply_fn({'params': params}, inputs, train=True,
                                    rngs={'dropout': dropout_key})
            
            if task == 'classification':
                loss = optax.sigmoid_binary_cross_entropy(logits, onehot(labels, num_output_classes)).mean()
            elif task == 'mlm':
                loss = cross_entropy_loss(logits, labels, num_classes=num_output_classes)
            else:
                raise ValueError("Task must be 'classification' or 'mlm'")
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(state, batch):
        inputs, labels = batch
        # Pass the input tensor directly to the model
        logits = state.apply_fn({'params': state.params}, inputs, train=False)
        
        if task == 'classification':
            loss = optax.sigmoid_binary_cross_entropy(logits, onehot(labels, num_output_classes)).mean()
            return loss, logits, labels
        elif task == 'mlm':
            loss = cross_entropy_loss(logits, labels, num_classes=num_output_classes)
            return loss, None, None # No predictions needed for PPL calculation
        else:
            raise ValueError("Task must be 'classification' or 'mlm'")

    best_val_metric = -1.0 if task == 'classification' else float('inf')
    best_epoch = 0
    best_state = state  # <<< --- ADD: Initialize best_state
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # --- TRAINING ---
        total_loss = 0
        num_batches = 0
        # Re-create the generator for each epoch
        pbar = tqdm(train_dataloader(), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            key, dropout_key = jax.random.split(key)
            state, loss = train_step(state, batch, dropout_key)
            total_loss += loss
            num_batches += 1
            
            if task == 'classification':
                 pbar.set_postfix(Loss=f"{loss:.4f}")
            elif task == 'mlm':
                ppl = jnp.exp(loss)
                pbar.set_postfix(Loss=f"{loss:.4f}", PPL=f"{ppl:.2f}")

        avg_train_loss = total_loss / num_batches
        
        # --- VALIDATION ---
        total_val_loss = 0
        num_val_batches = 0
        all_preds, all_labels = [], []
        pbar_val = tqdm(val_dataloader(), desc="Validation", leave=False)
        for batch in pbar_val:
            loss, preds, labels = eval_step(state, batch)
            total_val_loss += loss
            num_val_batches += 1
            if task == 'classification':
                all_preds.append(jax.nn.softmax(preds, axis=-1))
                all_labels.append(labels)

        avg_val_loss = total_val_loss / num_val_batches
        
        # --- METRIC CALCULATION & EARLY STOPPING ---
        if task == 'classification':
            val_preds = jnp.concatenate([p[:, 1] for p in all_preds])
            val_labels = jnp.concatenate(all_labels)
            val_auc = roc_auc_score(val_labels, val_preds)
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val AUC = {val_auc*100:.2f}%")
            
            if val_auc > best_val_metric:
                best_val_metric = val_auc
                best_epoch = epoch + 1
                best_state = state  # <<< --- ADD: Update best_state
        
        elif task == 'mlm':
            val_ppl = jnp.exp(avg_val_loss)
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val PPL = {val_ppl:.2f}")
            
            if avg_val_loss < best_val_metric:
                best_val_metric = avg_val_loss
                best_epoch = epoch + 1
                best_state = state  # <<< --- ADD: Update best_state

    total_training_time = time.time() - start_time
    metric_name = "best validation AUC" if task == 'classification' else "best validation loss"
    metric_value = f"{best_val_metric*100:.2f}%" if task == 'classification' else f"{best_val_metric:.4f}"
    print(f"Total training time = {total_training_time:.2f}s, {metric_name} = {metric_value} at epoch {best_epoch}")
    
    # --- TESTING ---
    # Use the best_state for testing
    total_test_loss = 0
    num_test_batches = 0
    all_preds, all_labels = [], []
    pbar_test = tqdm(test_dataloader(), desc="Testing")
    for batch in pbar_test:
        # <<< --- CHANGE: Use best_state, not final state --- >>>
        loss, preds, labels = eval_step(best_state, batch)
        total_test_loss += loss
        num_test_batches += 1
        if task == 'classification':
            all_preds.append(jax.nn.softmax(preds, axis=-1))
            all_labels.append(labels)

    avg_test_loss = total_test_loss / num_test_batches
    
    if task == 'classification':
        test_preds = jnp.concatenate([p[:, 1] for p in all_preds])
        test_labels = jnp.concatenate(all_labels)
        test_auc = roc_auc_score(test_labels, test_preds)
        print(f"Test Loss = {avg_test_loss:.4f}, Test AUC = {test_auc*100:.2f}%")
        # <<< --- CHANGE: Return best_state --- >>>
        return (avg_test_loss, test_auc, test_preds, test_labels), best_state
    elif task == 'mlm':
        test_ppl = jnp.exp(avg_test_loss)
        print(f"Test Loss = {avg_test_loss:.4f}, Test PPL = {test_ppl:.2f}")
        # <<< --- CHANGE: Return best_state --- >>>
        return (avg_test_loss, test_ppl), best_state

