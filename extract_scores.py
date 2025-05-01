#!/usr/bin/env python3
import os
import json
import argparse
import glob
import csv
import re
from pathlib import Path
try:
    import questionary
except ImportError:
    print("questionary not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "questionary"])
    import questionary

def extract_hyperparams_from_dirname(dirname):
    """
    Extract hyperparameters from directory name like 'run_1_batch_size_16_learning_rate_1e-7'
    
    Args:
        dirname (str): Directory name containing hyperparameters
        
    Returns:
        dict: Extracted hyperparameters
    """
    # Extract run number
    run_num = int(dirname.split('_')[1])
    
    # Extract hyperparameters using regex
    batch_size_match = re.search(r'batch_size_(\d+)', dirname)
    lr_match = re.search(r'learning_rate_([0-9e\-]+)', dirname)
    
    params = {
        'run': run_num,
        'batch_size': int(batch_size_match.group(1)) if batch_size_match else None,
        'learning_rate': float(lr_match.group(1)) if lr_match else None
    }
    
    return params

def extract_hypersearch_scores(base_dir, selected_runs=None):
    """
    Extract scores from hyperparameter search runs
    
    Args:
        base_dir (str): Path to the hypersearch directory
        selected_runs (list): List of run folders to process. If None, process all.
    
    Returns:
        list: List of dictionaries containing hyperparameters and metrics per epoch
    """
    results = []
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_') and os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter runs if a selection was provided
    if selected_runs:
        run_dirs = [d for d in run_dirs if d in selected_runs]
    
    for run_dir in sorted(run_dirs):
        run_path = os.path.join(base_dir, run_dir)
        
        # Extract hyperparameters from directory name
        hyperparams = extract_hyperparams_from_dirname(run_dir)
        
        # Read history.json for metrics
        history_file = os.path.join(run_path, 'history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Extract data for each epoch
            num_epochs = len(history_data['train_loss'])
            
            for epoch in range(num_epochs):
                epoch_data = {
                    'epoch': epoch + 1,
                    'train_loss': history_data['train_loss'][epoch],
                    'val_loss': history_data['val_loss'][epoch],
                    'val_accuracy': history_data['metrics']['accuracy'][epoch],
                    'val_f1_score': history_data['metrics']['f1_score'][epoch]
                }
                
                # Combine hyperparameters and metrics
                row = {**hyperparams, **epoch_data}
                results.append(row)
    
    return results

def extract_cv_scores(base_dir, selected_configs=None):
    """
    Extract scores from cross-validation runs
    
    Args:
        base_dir (str): Path to the base directory containing config folders
        selected_configs (list): List of config folders to process. If None, process all.
    
    Returns:
        dict: Structured data with all scores
    """
    results = {}
    
    # Find all config directories
    config_dirs = [d for d in os.listdir(base_dir) if d.startswith('config_') and os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter config dirs if a selection was provided
    if selected_configs:
        config_dirs = [d for d in config_dirs if d in selected_configs]
    
    for config_dir in sorted(config_dirs):
        config_path = os.path.join(base_dir, config_dir)
        results[config_dir] = {}
        
        # Find all fold directories
        fold_dirs = [d for d in os.listdir(config_path) if d.startswith('fold_') and os.path.isdir(os.path.join(config_path, d))]
        
        for fold_dir in sorted(fold_dirs, key=lambda x: int(x.split('_')[1])):
            fold_path = os.path.join(config_path, fold_dir)
            fold_idx = int(fold_dir.split('_')[1])
            
            # Check for history.json, which contains all metrics per epoch
            history_file = os.path.join(fold_path, 'history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Extract data for each epoch
                num_epochs = len(history_data['train_loss'])
                fold_results = []
                
                for epoch in range(num_epochs):
                    epoch_data = {
                        'epoch': epoch + 1,
                        'train_loss': history_data['train_loss'][epoch],
                        'val_loss': history_data['val_loss'][epoch],
                        'val_accuracy': history_data['metrics']['accuracy'][epoch],
                        'val_f1_score': history_data['metrics']['f1_score'][epoch]
                    }
                    fold_results.append(epoch_data)
                
                results[config_dir][f'fold_{fold_idx}'] = fold_results
    
    return results

def extract_config_hyperparams(config_dir):
    """
    Extract hyperparameters from a config directory's params.json file
    
    Args:
        config_dir (str): Path to the config directory
        
    Returns:
        dict: Extracted hyperparameters
    """
    params_file = os.path.join(config_dir, 'params.json')
    if not os.path.exists(params_file):
        return {'batch_size': None, 'learning_rate': None}
        
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Extract from original_run field (e.g. "run_13_batch_size_32_learning_rate_2e-5")
    run_name = params.get('original_run', '')
    
    # Extract hyperparameters using regex
    batch_size_match = re.search(r'batch_size_(\d+)', run_name)
    lr_match = re.search(r'learning_rate_([0-9e\-]+)', run_name)
    
    return {
        'batch_size': int(batch_size_match.group(1)) if batch_size_match else None,
        'learning_rate': float(lr_match.group(1)) if lr_match else None
    }

def cv_json_to_csv_rows(results, base_dir):
    """
    Convert the nested cross-validation JSON structure to flat CSV rows
    
    Args:
        results (dict): Structured data with all scores
        base_dir (str): Base directory containing the config folders
    
    Returns:
        list: List of dictionaries, each representing a row in the CSV
    """
    csv_rows = []
    
    for config, folds in results.items():
        # Extract config number from config_X
        config_num = config.split('_')[1]
        
        # Extract hyperparameters from the config directory
        config_dir = os.path.join(base_dir, config)
        hyperparams = extract_config_hyperparams(config_dir)
        
        for fold, epochs in folds.items():
            # Extract fold number from fold_X
            fold_num = fold.split('_')[1]
            
            for epoch_data in epochs:
                # Create a flat dictionary for this row
                row = {
                    'config': config_num,
                    'batch_size': hyperparams['batch_size'],
                    'learning_rate': hyperparams['learning_rate'],
                    'fold': fold_num,
                    'epoch': epoch_data['epoch']
                }
                # Add all metrics
                for metric, value in epoch_data.items():
                    if metric != 'epoch':  # Skip epoch as it's already added
                        row[metric] = value
                
                csv_rows.append(row)
    
    return csv_rows

def detect_run_type(run_dir):
    """
    Detect whether this is a cross-validation or hyperparameter search run
    
    Args:
        run_dir (str): Path to the run directory
        
    Returns:
        str: 'cv' or 'hypersearch'
    """
    # Check for config_X directories (CV) vs run_X directories (hypersearch)
    has_config_dirs = any(d.startswith('config_') for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)))
    has_run_dirs = any(d.startswith('run_') for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)))
    
    if has_config_dirs:
        return 'cv'
    elif has_run_dirs:
        return 'hypersearch'
    else:
        return None

def select_run_folder(runs_dir):
    """
    Show interactive select to choose which run folder to process
    
    Args:
        runs_dir (str): Path to the directory containing multiple run folders
        
    Returns:
        str: Selected run folder path
    """
    # Find all run directories (cv_ or hypersearch_)
    run_dirs = [d for d in os.listdir(runs_dir) 
               if (d.startswith('cv_') or d.startswith('hypersearch_')) 
               and os.path.isdir(os.path.join(runs_dir, d))]
    
    if not run_dirs:
        print(f"No run folders found in {runs_dir}")
        return None
    
    # Sort by date (newest first)
    run_dirs.sort(reverse=True)
    
    # Use questionary to prompt user for selection
    selected = questionary.select(
        'Select a run folder to process:',
        choices=run_dirs
    ).ask()
    
    return os.path.join(runs_dir, selected) if selected else None

def select_hypersearch_runs(base_dir):
    """
    Show interactive multiselect to choose which hyperparameter runs to process
    
    Args:
        base_dir (str): Path to the hypersearch directory
        
    Returns:
        list: Selected run folder names
    """
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_') and os.path.isdir(os.path.join(base_dir, d))]
    
    if not run_dirs:
        print(f"No run folders found in {base_dir}")
        return []
    
    # Use questionary to prompt user for selection
    message = 'Select runs to process (space to select, enter to confirm):'
    sorted_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[1]))  # Sort by run number
    selected = questionary.checkbox(
        message,
        choices=sorted_dirs,
    ).ask()
    
    return selected or []

def select_config_folders(base_dir):
    """
    Show interactive multiselect to choose which config folders to process
    
    Args:
        base_dir (str): Path to the base directory containing config folders
        
    Returns:
        list: Selected config folder names
    """
    # Find all config directories
    config_dirs = [d for d in os.listdir(base_dir) if d.startswith('config_') and os.path.isdir(os.path.join(base_dir, d))]
    
    if not config_dirs:
        print(f"No config folders found in {base_dir}")
        return []
    
    # Use questionary to prompt user for selection
    message = 'Select config folders to process (space to select, enter to confirm):'
    sorted_dirs = sorted(config_dirs)
    selected = questionary.checkbox(
        message,
        choices=sorted_dirs,
    ).ask()
    
    return selected or []

def select_output_format():
    """
    Show interactive select to choose output format(s)
    
    Returns:
        dict: Selected output formats {'json': bool, 'csv': bool}
    """
    choices = [
        {"name": "JSON", "value": "json"},
        {"name": "CSV", "value": "csv"},
        {"name": "Both JSON and CSV", "value": "both"}
    ]
    
    selected = questionary.select(
        'Select output format:',
        choices=choices
    ).ask()
    
    formats = {
        'json': selected in ['json', 'both'],
        'csv': selected in ['csv', 'both']
    }
    
    return formats

def main():
    parser = argparse.ArgumentParser(description='Extract scores from runs directory')
    parser.add_argument('--runs-dir', '-r', default='runs', help='Directory containing run folders')
    parser.add_argument('--output-name', '-o', default='scores', help='Base output filename (without extension)')
    parser.add_argument('--specific-run', '-s', help='Process a specific run directory instead of selecting')
    parser.add_argument('--all', '-a', action='store_true', help='Process all folders without selection prompt')
    args = parser.parse_args()
    
    # Determine the run directory to process
    run_dir = None
    if args.specific_run:
        run_dir = args.specific_run
        if not os.path.isdir(run_dir):
            print(f"Error: {run_dir} is not a valid directory")
            return
    else:
        if not os.path.isdir(args.runs_dir):
            print(f"Error: {args.runs_dir} is not a valid directory")
            return
        
        run_dir = select_run_folder(args.runs_dir)
        if not run_dir:
            print("No run folder selected. Exiting.")
            return
    
    # Detect run type
    run_type = detect_run_type(run_dir)
    if not run_type:
        print(f"Error: Could not determine run type for {run_dir}")
        return
    
    # Extract scores based on run type
    if run_type == 'cv':
        # Select config folders to process
        selected_configs = None if args.all else select_config_folders(run_dir)
        if not args.all and not selected_configs:
            print("No config folders selected. Exiting.")
            return
        
        # Extract scores
        results = extract_cv_scores(run_dir, selected_configs)
        csv_rows = cv_json_to_csv_rows(results, run_dir)
    else:  # hypersearch
        # Select runs to process
        selected_runs = None if args.all else select_hypersearch_runs(run_dir)
        if not args.all and not selected_runs:
            print("No runs selected. Exiting.")
            return
        
        # Extract scores
        results = extract_hypersearch_scores(run_dir, selected_runs)
        csv_rows = results  # Already in flat format
    
    # Select output format
    output_formats = select_output_format()
    
    # Determine output base name (without extension)
    output_base = args.output_name
    
    # Save JSON if requested
    if output_formats['json']:
        json_path = os.path.join(run_dir, f"{output_base}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Scores extracted and saved to {json_path}")
    
    # Save CSV if requested
    if output_formats['csv']:
        if csv_rows:
            csv_path = os.path.join(run_dir, f"{output_base}.csv")
            with open(csv_path, 'w', newline='') as f:
                # Get fieldnames from the first row
                fieldnames = list(csv_rows[0].keys())
                
                # Move important fields to the front
                if run_type == 'cv':
                    front_fields = ['epoch', 'fold', 'config']
                else:
                    front_fields = ['epoch', 'run', 'batch_size', 'learning_rate']
                
                for field in reversed(front_fields):
                    if field in fieldnames:
                        fieldnames.remove(field)
                        fieldnames.insert(0, field)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"CSV data saved to {csv_path}")
        else:
            print("No data to write to CSV.")

if __name__ == '__main__':
    main() 