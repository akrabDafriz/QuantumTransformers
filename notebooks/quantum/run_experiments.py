import argparse
import os
import jax
import torch 
import numpy as np
import pickle
from quantum_transformers.datasets import get_custom_classification_dataloader
from quantum_transformers.transformers import Transformer
from quantum_transformers.quantum_layer import (
    get_circuit, 
    basic_vqc, 
    vqc_ry_cnot, 
    vqc_h_rx_cnot, 
    vqc_all_rot_cnot, 
    vqc_rx_crx
)
from quantum_transformers.training import train_and_evaluate

# --- Configuration Constants ---

HIDDEN_SIZE = 8       # Updated to 8 as per your request       
NUM_HEADS = 2
NUM_BLOCKS = 2
MLP_HIDDEN = 8        # Updated to 8 to match HIDDEN_SIZE (prevents shape mismatch)        
MAX_SEQ_LEN = 32
BATCH_SIZE = 8        # <--- REDUCED from 16 to 8 to prevent OOM Crashes        
NUM_EPOCHS = 40       
LEARNING_RATE = 1e-3
NUM_LAYERS_VQC = 2

DATA_PATHS = {
    'mc': {
        'type': 'mc_rp',
        'train': 'data/mc_train_data.txt',
        'val': 'data/mc_dev_data.txt',
        'test': 'data/mc_test_data.txt',
        'tokenizer': 'wordlevel',
        'vocab': 17,  
        'shared_files': None
    },
    'rp': {
        'type': 'mc_rp',
        'train': 'data/rp_train_data.txt',
        'val': None,
        'test': 'data/rp_test_data.txt',
        'tokenizer': 'wordlevel',
        'vocab': 115, 
        'shared_files': None
    },
    'imdb': {
        'type': 'sentiment',
        'files': ['data/imdb_labelled.txt'],
        'tokenizer': 'bpe',
        'vocab': 1000,
        'shared_files': True
    },
    'amazon': {
        'type': 'sentiment',
        'files': ['data/amazon_cells_labelled.txt'],
        'tokenizer': 'bpe',
        'vocab': 1000,
        'shared_files': True
    },
    'yelp': {
        'type': 'sentiment',
        'files': ['data/yelp_labelled.txt'],
        'tokenizer': 'bpe',
        'vocab': 1000,
        'shared_files': True
    }
}

SENTIMENT_FILES = [
    'data/imdb_labelled.txt', 
    'data/amazon_cells_labelled.txt', 
    'data/yelp_labelled.txt'
]

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Experiment Definitions ---

def get_experiment_config(experiment_id):
    if experiment_id == 1:
        return [
            ("Classical", None, None), 
            ("Basic VQC", basic_vqc, (NUM_LAYERS_VQC,))
        ]
    elif experiment_id == 2:
        return [
            ("Design 1 (RY+CNOT)", vqc_ry_cnot, (NUM_LAYERS_VQC,)),
            ("Design 3 (Full Rot+CNOT)", vqc_all_rot_cnot, (NUM_LAYERS_VQC, 3))
        ]
    elif experiment_id == 3:
        return [("Design 2 (H+RX+CNOT)", vqc_h_rx_cnot, (NUM_LAYERS_VQC,))]
    elif experiment_id == 4:
        return [("Design 4 (RX+CRX)", vqc_rx_crx, (NUM_LAYERS_VQC, 2))]
    else:
        raise ValueError("Invalid Experiment ID. Choose 1-4.")


def run_single_trial(dataset_name, dataset_config, vqc_name, vqc_func, w_shape, exp_id, trial_num, batch_size):
    print(f"\n{'='*60}")
    print(f"Running {vqc_name} on {dataset_name.upper()} Dataset | Trial {trial_num}")
    print(f"{'='*60}")
    
    torch.manual_seed(trial_num)

    if dataset_config.get('shared_files') is True:
        tokenizer_files = SENTIMENT_FILES
    else:
        tokenizer_files = None 

    # 1. Load Data
    print(f"Loading Datasets (Batch Size: {batch_size})...")
    try:
        if dataset_config['type'] == 'sentiment':
            train_loader, val_loader, test_loader, tokenizer = get_custom_classification_dataloader(
                dataset_type='sentiment',
                file_paths=dataset_config['files'],
                batch_size=batch_size, # Use dynamic batch size
                max_seq_len=MAX_SEQ_LEN,
                tokenizer_type=dataset_config['tokenizer'], 
                vocab_size=dataset_config['vocab'],         
                tokenizer_files=tokenizer_files             
            )
        else:
            train_loader, val_loader, test_loader, tokenizer = get_custom_classification_dataloader(
                dataset_type='mc_rp',
                train_path=dataset_config['train'],
                val_path=dataset_config['val'],
                test_path=dataset_config['test'],
                batch_size=batch_size, # Use dynamic batch size
                max_seq_len=MAX_SEQ_LEN,
                tokenizer_type=dataset_config['tokenizer'], 
                vocab_size=dataset_config['vocab'],         
                tokenizer_files=tokenizer_files             
            )
    except Exception as e:
        print(f"SKIPPING: Could not load data for {dataset_name}. Error: {e}")
        return None

    # 2. Initialize Model
    print(f"Initializing Transformer with {vqc_name}...")
    
    if vqc_func is not None:
        circuit_fn = get_circuit(vqc=vqc_func)
        current_w_shape = w_shape
    else:
        circuit_fn = None
        current_w_shape = (1,)

    model = Transformer(
        num_tokens=tokenizer.get_vocab_size(),
        max_seq_len=MAX_SEQ_LEN,
        num_classes=2,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_BLOCKS,
        mlp_hidden_size=MLP_HIDDEN,
        dropout=0.1,
        quantum_w_shape=current_w_shape,
        quantum_attn_circuit=circuit_fn,
        quantum_mlp_circuit=circuit_fn
    )

    # 3. Train
    print("Starting Training...")
    (test_loss, test_acc), best_state, history = train_and_evaluate(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        task='classification',
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=trial_num 
    )
    
    print(f"RESULT: {dataset_name} | {vqc_name} | Run {trial_num} | Test Acc: {test_acc:.4f}")

    # 4. Save History
    safe_vqc_name = vqc_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
    save_filename = f"Exp{exp_id}_{dataset_name}_{safe_vqc_name}_run{trial_num}.pkl"
    save_path = os.path.join(RESULTS_DIR, save_filename)
    
    with open(save_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"History saved to {save_path}")

    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Run Quantum Transformer Experiments")
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4], help="Experiment ID (1-4).")
    parser.add_argument('--dataset', type=str, choices=['mc', 'rp', 'imdb', 'amazon', 'yelp'], help="Specific dataset.")
    # Add flag to control batch size manually
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f"Batch size (Default: {BATCH_SIZE})")
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else DATA_PATHS.keys()
    experiments_to_run = [args.exp] if args.exp else [1, 2, 3, 4]

    results = {}

    for exp_id in experiments_to_run:
        print(f"\n\n################################################")
        print(f"### STARTING EXPERIMENT {exp_id} ###")
        print(f"################################################")
        
        configs = get_experiment_config(exp_id)
        
        for vqc_name, vqc_func, w_shape in configs:
            for ds_name in datasets_to_run:
                
                if ds_name in ['mc', 'rp']:
                    num_trials = 5
                else:
                    num_trials = 3
                
                print(f"\n>>> Running {num_trials} trials for {ds_name}...")
                
                for trial in range(num_trials):
                    key = f"Exp{exp_id}_{ds_name}_{vqc_name}_run{trial}"
                    # Pass batch_size from args
                    acc = run_single_trial(ds_name, DATA_PATHS[ds_name], vqc_name, vqc_func, w_shape, exp_id, trial, args.batch_size)
                    results[key] = acc

    print("\n\n" + "="*30)
    print("FINAL SUMMARY RESULTS")
    print("="*30)
    for k, v in results.items():
        val = f"{v:.4f}" if v is not None else "Failed/Skipped"
        print(f"{k}: {val}")


if __name__ == '__main__':
    print(f"JAX Devices: {jax.devices()}")
    main()