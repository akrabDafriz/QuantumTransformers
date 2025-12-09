import jax
import jax.numpy as jnp
import numpy as np  # Added for CPU array conversion
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
    
    weights = jnp.where(labels_flat != -100, 1.0, 0.0)
    
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_flat,
        labels=jnp.where(labels_flat != -100, labels_flat, 0)
    )
    
    losses = losses * weights
    return jnp.sum(losses) / (jnp.sum(weights) + 1e-8)

def create_train_state(rng, model, sample_input, learning_rate):
    """Creates initial TrainState."""
    params = model.init(rng, sample_input, train=True)['params']
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, dropout_rng, num_classes):
    """Runs a single training step."""
    inputs, targets = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs, train=True, rngs={'dropout': dropout_rng})
        if logits.ndim == 3: # MLM
             loss = cross_entropy_loss(logits, targets, num_classes)
        else: # Classification
             loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

@jax.jit
def eval_step(state, batch):
    """Runs a single evaluation step."""
    inputs, targets = batch
    logits = state.apply_fn({'params': state.params}, inputs, train=False)
    
    if logits.ndim == 3: # MLM
        num_classes = logits.shape[-1]
        loss = cross_entropy_loss(logits, targets, num_classes)
    else: # Classification
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        
    return loss, logits, targets

def train_and_evaluate(model, train_dataloader, val_dataloader, test_dataloader, 
                       task='classification', num_epochs=10, learning_rate=1e-3, num_classes=None, seed=0):
    
    # Use the provided seed for reproducibility per trial
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    sample_batch = next(iter(train_dataloader))
    sample_input = jnp.array(sample_batch[0])
    state = create_train_state(init_rng, model, sample_input, learning_rate)
    
    best_val_metric = -float('inf') if task == 'classification' else float('inf')
    best_state = state
    best_epoch = 0
    
    # 1. Recording containers
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metric': [],
        'test_preds': None,   
        'test_labels': None,
        'seed': seed  # Record seed used
    }

    print(f"Starting training for {num_epochs} epochs (Seed: {seed})...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # --- TRAINING ---
        total_train_loss = 0
        num_train_batches = 0
        
        pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in pbar_train:
            batch = (jnp.array(batch[0]), jnp.array(batch[1]))
            dropout_rng, rng = jax.random.split(rng)
            state, loss, _ = train_step(state, batch, dropout_rng, num_classes)
            total_train_loss += loss
            num_train_batches += 1
            pbar_train.set_postfix({'loss': float(loss)})
            
        avg_train_loss = total_train_loss / num_train_batches
        history['train_loss'].append(float(avg_train_loss))
        
        # --- VALIDATION ---
        if val_dataloader is not None:
            total_val_loss = 0
            num_val_batches = 0
            all_preds, all_labels = [], []
            
            pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for batch in pbar_val:
                batch = (jnp.array(batch[0]), jnp.array(batch[1]))
                loss, preds, labels = eval_step(state, batch)
                total_val_loss += loss
                num_val_batches += 1
                
                if task == 'classification':
                    all_preds.append(jax.nn.softmax(preds, axis=-1))
                    all_labels.append(labels)

            avg_val_loss = total_val_loss / num_val_batches
            history['val_loss'].append(float(avg_val_loss))
            
            if task == 'classification':
                val_preds = jnp.concatenate([p[:, 1] for p in all_preds])
                val_labels = jnp.concatenate(all_labels)
                try:
                    current_val_metric = float(roc_auc_score(val_labels, val_preds))
                except ValueError:
                     current_val_metric = 0.5 
                metric_name = "AUC"
                is_better = current_val_metric > best_val_metric
            else: 
                current_val_metric = float(jnp.exp(avg_val_loss))
                metric_name = "PPL"
                is_better = current_val_metric < best_val_metric

            history['val_metric'].append(current_val_metric)
            
            val_str = f"Val Loss: {avg_val_loss:.4f}, Val {metric_name}: {current_val_metric:.4f}"
            
            if is_better:
                best_val_metric = current_val_metric
                best_state = state
                best_epoch = epoch + 1
        else:
            history['val_loss'].append(None)
            history['val_metric'].append(None)
            metric_name = "N/A"
            val_str = "Val: N/A"
            best_state = state
            best_epoch = epoch + 1

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | {val_str}")

    total_training_time = time.time() - start_time
    
    if val_dataloader is not None:
        metric_value = f"{best_val_metric*100:.2f}%" if task == 'classification' else f"{best_val_metric:.4f}"
        print(f"Total training time = {total_training_time:.2f}s, Best {metric_name} = {metric_value} at epoch {best_epoch}")
    else:
        print(f"Total training time = {total_training_time:.2f}s, Validation skipped.")

    # --- TESTING ---
    if test_dataloader is not None:
        total_test_loss = 0
        num_test_batches = 0
        all_preds, all_labels = [], []
        pbar_test = tqdm(test_dataloader, desc="Testing", leave=False)
        for batch in pbar_test:
            batch = (jnp.array(batch[0]), jnp.array(batch[1]))
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
            
            # Save raw predictions for ROC plotting later
            history['test_preds'] = np.array(test_preds)
            history['test_labels'] = np.array(test_labels)

            try:
                test_auc = roc_auc_score(test_labels, test_preds)
                print(f"Test Loss = {avg_test_loss:.4f}, Test AUC = {test_auc*100:.2f}%")
                return (avg_test_loss, test_auc), best_state, history
            except ValueError:
                print(f"Test Loss = {avg_test_loss:.4f}, Test AUC = N/A (Error)")
                return (avg_test_loss, 0.0), best_state, history
        else:
            test_ppl = jnp.exp(avg_test_loss)
            print(f"Test Loss = {avg_test_loss:.4f}, Test PPL = {test_ppl:.4f}")
            return (avg_test_loss, test_ppl), best_state, history
    else:
        print("No test set provided.")
        return (None, None), best_state, history