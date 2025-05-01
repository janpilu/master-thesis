#!/usr/bin/env python
"""Interactive CLI entrypoint for hyperparameter search and model training."""

import os
import sys
import subprocess
import questionary
from pathlib import Path
import logging
import json
from typing import List, Dict, Any, Optional
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_config_files() -> List[Path]:
    """Get all configuration files in the config directory."""
    config_dir = Path("config")
    if not config_dir.exists():
        return []
    
    return sorted(config_dir.glob("*.yaml"))

def get_hypersearch_dirs() -> List[Path]:
    """Get all hyperparameter search directories."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    return sorted([d for d in runs_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("hypersearch_")], 
                  key=lambda x: x.stat().st_mtime, reverse=True)

def get_incomplete_runs(hypersearch_dir: Path) -> List[Path]:
    """Find runs that did not complete successfully."""
    if not hypersearch_dir.exists():
        return []
    
    # Get all run directories
    run_dirs = [d for d in hypersearch_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    incomplete_runs = []
    
    for run_dir in run_dirs:
        # Check if the run has a training.log file with completion message
        log_file = run_dir / "training.log"
        completed = False
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    if "Training completed in" in log_content:
                        completed = True
                        logger.info(f"Run {run_dir.name} has completion message")
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        
        # A run is incomplete if it doesn't have a completion message in the log
        if not completed:
            incomplete_runs.append(run_dir)
            logger.info(f"Run {run_dir.name} is incomplete: no completion message in log")
    
    return sorted(incomplete_runs, key=lambda x: int(x.name.split('_')[1]))

def get_run_info(run_dir: Path) -> Dict[str, Any]:
    """Get information about a run from its parameters file."""
    run_params_file = run_dir / "run_params.json"
    if not run_params_file.exists():
        return {"name": run_dir.name, "parameters": {}}
    
    try:
        with open(run_params_file, 'r', encoding='utf-8') as f:
            run_params = json.load(f)
            return {
                "name": run_dir.name,
                "parameters": run_params.get("parameters", {}),
                "run_index": run_params.get("run_index", 0),
                "total_runs": run_params.get("total_runs", 0),
                "random_seed": run_params.get("random_seed", None)
            }
    except Exception as e:
        logger.error(f"Error reading run parameters file {run_params_file}: {e}")
        return {"name": run_dir.name, "parameters": {}}

def format_run_choice(run_dir: Path) -> str:
    """Format a run directory as a choice for questionary."""
    run_info = get_run_info(run_dir)
    params_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in run_info.get("parameters", {}).items()])
    return f"{run_dir.name} ({params_str})"

def start_new_run():
    """Start a new hyperparameter search run."""
    # Get available config files
    config_files = get_config_files()
    if not config_files:
        print("No configuration files found in the config directory.")
        return
    
    # Ask user to select a config file
    config_choices = [f.name for f in config_files]
    config_file = questionary.select(
        "Select a configuration file:",
        choices=config_choices
    ).ask()
    
    if not config_file:
        return
    
    config_path = Path("config") / config_file
    
    # Ask if user wants to specify parameter variations
    use_param_variations = questionary.confirm(
        "Do you want to specify parameter variations from the command line?",
        default=False
    ).ask()
    
    param_variations = []
    if use_param_variations:
        while True:
            param_path = questionary.text(
                "Enter parameter path (e.g., 'training.learning_rate') or leave empty to finish:"
            ).ask()
            
            if not param_path:
                break
            
            param_values = questionary.text(
                "Enter comma-separated values (e.g., '1e-3,1e-4,1e-5'):"
            ).ask()
            
            if param_values:
                param_variations.append(f"{param_path}={param_values}")
    
    # Ask if user wants to execute the runs
    execute_runs = questionary.confirm(
        "Do you want to execute the runs?",
        default=True
    ).ask()
    
    # Build command
    cmd = ["python", "hyper_main.py", "--config", str(config_path)]
    
    if param_variations:
        cmd.extend(["--param-variations"] + param_variations)
    
    if not execute_runs:
        cmd.append("--no-execute")
    
    # Execute command
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def continue_run():
    """Continue an existing hyperparameter search run."""
    # Get available hypersearch directories
    hypersearch_dirs = get_hypersearch_dirs()
    if not hypersearch_dirs:
        print("No hyperparameter search directories found.")
        return
    
    # Ask user to select a hypersearch directory
    hypersearch_choices = [d.name for d in hypersearch_dirs]
    hypersearch_dir_name = questionary.select(
        "Select a hyperparameter search directory:",
        choices=hypersearch_choices
    ).ask()
    
    if not hypersearch_dir_name:
        return
    
    hypersearch_dir = Path("runs") / hypersearch_dir_name
    
    # Get incomplete runs
    incomplete_runs = get_incomplete_runs(hypersearch_dir)
    if not incomplete_runs:
        print(f"No incomplete runs found in {hypersearch_dir}.")
        return
    
    # Show number of incomplete runs and ask for confirmation
    print(f"\nFound {len(incomplete_runs)} incomplete runs:")
    for run_dir in incomplete_runs:
        run_info = get_run_info(run_dir)
        params_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in run_info.get("parameters", {}).items()])
        print(f"- {run_dir.name} ({params_str})")
    
    proceed = questionary.confirm(
        f"\nDo you want to continue with these {len(incomplete_runs)} runs?",
        default=True
    ).ask()
    
    if not proceed:
        return
    
    # Ask if user wants to restart the last run
    restart_last = questionary.confirm(
        "Do you want to restart the incomplete runs (deletes and recreates the run directories)?",
        default=False
    ).ask()
    
    # Build command
    cmd = ["python", "hyper_main.py", "--continue-from", str(hypersearch_dir)]
    
    if restart_last:
        cmd.append("--restart-last")
    
    # Execute command
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def evaluate_misclassifications():
    """Evaluate misclassifications from a trained model."""
    # Check if evaluate_misclassifications.py exists
    eval_script_path = Path("evaluate_misclassifications.py")
    if not eval_script_path.exists():
        print("Evaluation script (evaluate_misclassifications.py) not found.")
        return
    
    # Get available hypersearch directories
    hypersearch_dirs = get_hypersearch_dirs()
    if not hypersearch_dirs:
        print("No hyperparameter search directories found.")
        return
    
    # Ask user to select a hypersearch directory
    hypersearch_choices = [d.name for d in hypersearch_dirs]
    hypersearch_dir_name = questionary.select(
        "Select a hyperparameter search directory:",
        choices=hypersearch_choices
    ).ask()
    
    if not hypersearch_dir_name:
        return
    
    hypersearch_dir = Path("runs") / hypersearch_dir_name
    
    # Get all run directories
    run_dirs = [d for d in hypersearch_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        print(f"No run directories found in {hypersearch_dir}.")
        return
    
    # Filter runs that have a best model
    completed_runs = []
    for run_dir in run_dirs:
        best_model_path = run_dir / "best-model.pt"
        if best_model_path.exists():
            completed_runs.append(run_dir)
    
    if not completed_runs:
        print(f"No completed runs with best models found in {hypersearch_dir}.")
        return
    
    # Format run choices
    run_choices = [format_run_choice(run_dir) for run_dir in completed_runs]
    
    # Ask user to select a run
    run_choice = questionary.select(
        "Select a run to evaluate:",
        choices=run_choices
    ).ask()
    
    if not run_choice:
        return
    
    # Extract run directory from choice
    selected_run_name = run_choice.split(" ")[0]
    selected_run_dir = hypersearch_dir / selected_run_name
    
    # Get the model path
    model_path = selected_run_dir / "best-model.pt"
    
    # Get the config path
    config_path = selected_run_dir / "config.yaml"
    
    # Execute command
    cmd = ["python", "evaluate_misclassifications.py", "--model", str(model_path), "--config", str(config_path)]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def annotation_tool():
    """Launch the annotation tool."""
    # Check if frontend.py exists
    frontend_path = Path("frontend.py")
    if not frontend_path.exists():
        print("Annotation tool (frontend.py) not found.")
        return
    
    # Execute command
    cmd = ["streamlit", "run", "frontend.py"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def view_results():
    """View results of completed runs."""
    # Get available hypersearch directories
    hypersearch_dirs = get_hypersearch_dirs()
    if not hypersearch_dirs:
        print("No hyperparameter search directories found.")
        return
    
    # Ask user to select a hypersearch directory
    hypersearch_choices = [d.name for d in hypersearch_dirs]
    hypersearch_dir_name = questionary.select(
        "Select a hyperparameter search directory:",
        choices=hypersearch_choices
    ).ask()
    
    if not hypersearch_dir_name:
        return
    
    hypersearch_dir = Path("runs") / hypersearch_dir_name
    
    # Get all run directories
    run_dirs = [d for d in hypersearch_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        print(f"No run directories found in {hypersearch_dir}.")
        return
    
    # Filter runs that have a history.json file
    completed_runs = []
    for run_dir in run_dirs:
        history_path = run_dir / "history.json"
        if history_path.exists():
            completed_runs.append(run_dir)
    
    if not completed_runs:
        print(f"No completed runs with history found in {hypersearch_dir}.")
        return
    
    # Format run choices
    run_choices = [format_run_choice(run_dir) for run_dir in completed_runs]
    
    # Ask user to select a run
    run_choice = questionary.select(
        "Select a run to view results:",
        choices=run_choices
    ).ask()
    
    if not run_choice:
        return
    
    # Extract run directory from choice
    selected_run_name = run_choice.split(" ")[0]
    selected_run_dir = hypersearch_dir / selected_run_name
    
    # Get the history file
    history_path = selected_run_dir / "history.json"
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Display results
        print("\nRun Results:")
        print(f"Run Directory: {selected_run_dir}")
        
        # Get run parameters
        run_info = get_run_info(selected_run_dir)
        print("\nParameters:")
        for k, v in run_info.get("parameters", {}).items():
            print(f"  {k.split('.')[-1]}: {v}")
        
        # Display training history
        print("\nTraining History:")
        print(f"  Epochs completed: {len(history.get('train_loss', []))}")
        
        if history.get('val_loss'):
            best_epoch = history['val_loss'].index(min(history['val_loss']))
            print(f"  Best epoch: {best_epoch + 1}")
            print(f"  Best validation loss: {min(history['val_loss']):.4f}")
        
        # Display metrics
        if 'metrics' in history:
            print("\nMetrics:")
            for metric_name, values in history['metrics'].items():
                if values:
                    print(f"  Best {metric_name}: {max(values):.4f}")
        
        # Ask if user wants to open the plots
        show_plots = questionary.confirm(
            "Do you want to open the plots?",
            default=True
        ).ask()
        
        if show_plots:
            # Check which plots exist
            loss_plot_path = selected_run_dir / "loss_plot.png"
            metrics_plot_path = selected_run_dir / "metrics_plot.png"
            
            if loss_plot_path.exists():
                # Use the default image viewer to open the plot
                subprocess.run(["open", str(loss_plot_path)])
            
            if metrics_plot_path.exists():
                subprocess.run(["open", str(metrics_plot_path)])
    
    except Exception as e:
        print(f"Error reading history file: {e}")

def rank_runs():
    """Rank all runs by their performance metrics."""
    # Get available hypersearch directories
    hypersearch_dirs = get_hypersearch_dirs()
    if not hypersearch_dirs:
        print("No hyperparameter search directories found.")
        return
    
    # Ask user to select a hypersearch directory
    hypersearch_choices = [d.name for d in hypersearch_dirs]
    hypersearch_dir_name = questionary.select(
        "Select a hyperparameter search directory:",
        choices=hypersearch_choices
    ).ask()
    
    if not hypersearch_dir_name:
        return
    
    hypersearch_dir = Path("runs") / hypersearch_dir_name
    
    # Get all run directories
    run_dirs = [d for d in hypersearch_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        print(f"No run directories found in {hypersearch_dir}.")
        return
    
    # Collect results from all runs with history.json
    run_results = []
    for run_dir in run_dirs:
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Get run parameters
            run_info = get_run_info(run_dir)
            
            # Extract best metrics
            metrics = {}
            if history.get('val_loss'):
                best_val_loss = min(history['val_loss'])
                best_epoch = history['val_loss'].index(best_val_loss)
                metrics['val_loss'] = best_val_loss
                metrics['best_epoch'] = best_epoch + 1
            
            if 'metrics' in history:
                for metric_name, values in history['metrics'].items():
                    if values:
                        if metric_name == 'loss':
                            metrics[metric_name] = min(values)
                        else:
                            metrics[metric_name] = max(values)
            
            # Add to results
            run_results.append({
                'run_dir': run_dir,
                'run_name': run_dir.name,
                'parameters': run_info.get('parameters', {}),
                'metrics': metrics,
                'epochs_completed': len(history.get('train_loss', []))
            })
        except Exception as e:
            logger.error(f"Error reading history for {run_dir}: {e}")
    
    if not run_results:
        print(f"No runs with history found in {hypersearch_dir}.")
        return
    
    # Ask user which metric to rank by
    available_metrics = set()
    for result in run_results:
        available_metrics.update(result['metrics'].keys())
    
    if not available_metrics:
        print("No metrics found in any run.")
        return
    
    metric_choices = list(available_metrics)
    if 'val_loss' in metric_choices:
        # Move val_loss to the front as it's commonly used
        metric_choices.remove('val_loss')
        metric_choices.insert(0, 'val_loss')
    
    rank_metric = questionary.select(
        "Select a metric to rank by:",
        choices=metric_choices
    ).ask()
    
    if not rank_metric:
        return
    
    # Filter runs that have the selected metric
    filtered_results = [r for r in run_results if rank_metric in r['metrics']]
    
    if not filtered_results:
        print(f"No runs found with metric '{rank_metric}'.")
        return
    
    # Sort runs by the selected metric
    reverse_sort = rank_metric != 'val_loss' and rank_metric != 'loss'
    sorted_results = sorted(
        filtered_results, 
        key=lambda x: x['metrics'].get(rank_metric, float('inf') if not reverse_sort else float('-inf')),
        reverse=reverse_sort
    )
    
    # Display ranked results
    print(f"\nRuns ranked by {rank_metric} ({'highest' if reverse_sort else 'lowest'} is best):")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results):
        params_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in result['parameters'].items()])
        metric_value = result['metrics'].get(rank_metric)
        
        print(f"{i+1}. {result['run_name']} - {rank_metric}: {metric_value:.4f}")
        print(f"   Parameters: {params_str}")
        print(f"   Epochs completed: {result['epochs_completed']}")
        
        # Show other metrics
        other_metrics = [m for m in result['metrics'] if m != rank_metric and m != 'best_epoch']
        if other_metrics:
            metrics_str = ", ".join([f"{m}: {result['metrics'][m]:.4f}" for m in other_metrics])
            print(f"   Other metrics: {metrics_str}")
        
        print("-" * 80)
    
    # Ask if user wants to view details of a specific run
    view_details = questionary.confirm(
        "Do you want to view details of a specific run?",
        default=True
    ).ask()
    
    if view_details:
        run_choices = [f"{i+1}. {r['run_name']}" for i, r in enumerate(sorted_results)]
        selected_run = questionary.select(
            "Select a run to view details:",
            choices=run_choices
        ).ask()
        
        if selected_run:
            # Extract run index from selection
            run_index = int(selected_run.split('.')[0]) - 1
            selected_run_dir = sorted_results[run_index]['run_dir']
            
            # Use the existing view_results function to show details
            # We need to simulate the selection process
            # First, get the history file
            history_path = selected_run_dir / "history.json"
            
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Display results
                print("\nRun Results:")
                print(f"Run Directory: {selected_run_dir}")
                
                # Get run parameters
                run_info = get_run_info(selected_run_dir)
                print("\nParameters:")
                for k, v in run_info.get("parameters", {}).items():
                    print(f"  {k.split('.')[-1]}: {v}")
                
                # Display training history
                print("\nTraining History:")
                print(f"  Epochs completed: {len(history.get('train_loss', []))}")
                
                if history.get('val_loss'):
                    best_epoch = history['val_loss'].index(min(history['val_loss']))
                    print(f"  Best epoch: {best_epoch + 1}")
                    print(f"  Best validation loss: {min(history['val_loss']):.4f}")
                
                # Display metrics
                if 'metrics' in history:
                    print("\nMetrics:")
                    for metric_name, values in history['metrics'].items():
                        if values:
                            print(f"  Best {metric_name}: {max(values):.4f}")
                
                # Ask if user wants to open the plots
                show_plots = questionary.confirm(
                    "Do you want to open the plots?",
                    default=True
                ).ask()
                
                if show_plots:
                    # Check which plots exist
                    loss_plot_path = selected_run_dir / "loss_plot.png"
                    metrics_plot_path = selected_run_dir / "metrics_plot.png"
                    
                    if loss_plot_path.exists():
                        # Use the default image viewer to open the plot
                        subprocess.run(["open", str(loss_plot_path)])
                    
                    if metrics_plot_path.exists():
                        subprocess.run(["open", str(metrics_plot_path)])
            
            except Exception as e:
                print(f"Error reading history file: {e}")

def start_cross_validation():
    """Start a cross-validation run using the best configurations from a hyperparameter search."""
    # Get available hypersearch directories
    hypersearch_dirs = get_hypersearch_dirs()
    if not hypersearch_dirs:
        print("No hyperparameter search directories found.")
        return
    
    # Ask user to select a hypersearch directory
    hypersearch_choices = [d.name for d in hypersearch_dirs]
    hypersearch_dir_name = questionary.select(
        "Select a hyperparameter search directory:",
        choices=hypersearch_choices
    ).ask()
    
    if not hypersearch_dir_name:
        return
    
    hypersearch_dir = Path("runs") / hypersearch_dir_name
    
    # Collect results from all runs with history.json
    run_results = []
    for run_dir in sorted(hypersearch_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
            
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Get run parameters
            run_info = get_run_info(run_dir)
            
            # Extract best metrics
            metrics = {}
            if history.get('val_loss'):
                best_val_loss = min(history['val_loss'])
                metrics['val_loss'] = best_val_loss
            
            if 'metrics' in history:
                for metric_name, values in history['metrics'].items():
                    if values:
                        if metric_name == 'loss':
                            metrics[metric_name] = min(values)
                        else:
                            metrics[metric_name] = max(values)
            
            config_path = run_dir / "config.yaml"
            
            # Add to results
            run_results.append({
                'run_dir': run_dir,
                'run_name': run_dir.name,
                'parameters': run_info.get('parameters', {}),
                'metrics': metrics,
                'config_path': config_path,
            })
        except Exception as e:
            logger.error(f"Error reading history for {run_dir}: {e}")
    
    if not run_results:
        print(f"No runs with history found in {hypersearch_dir}.")
        return
    
    # Ask user which metric to rank by
    available_metrics = set()
    for result in run_results:
        available_metrics.update(result['metrics'].keys())
    
    metric_choices = list(available_metrics)
    if 'val_loss' in metric_choices:
        # Move val_loss to the front as it's commonly used
        metric_choices.remove('val_loss')
        metric_choices.insert(0, 'val_loss')
    
    rank_metric = questionary.select(
        "Select a metric to rank by:",
        choices=metric_choices
    ).ask()
    
    if not rank_metric:
        return
    
    # Filter runs that have the selected metric
    filtered_results = [r for r in run_results if rank_metric in r['metrics']]
    
    if not filtered_results:
        print(f"No runs found with metric '{rank_metric}'.")
        return
    
    # Sort runs by the selected metric
    reverse_sort = rank_metric != 'val_loss' and rank_metric != 'loss'
    sorted_results = sorted(
        filtered_results, 
        key=lambda x: x['metrics'].get(rank_metric, float('inf') if not reverse_sort else float('-inf')),
        reverse=reverse_sort
    )
    
    # Get the top N runs
    num_configs = questionary.select(
        "How many top configurations would you like to use?",
        choices=["1", "2", "3", "5", "10"]
    ).ask()
    
    if not num_configs:
        return
    
    num_configs = int(num_configs)
    top_runs = sorted_results[:num_configs]
    
    # Ask for number of folds
    n_folds = questionary.select(
        "Select number of folds for cross-validation:",
        choices=["5", "10"]
    ).ask()
    
    if not n_folds:
        return
    
    n_folds = int(n_folds)
    
    # Ask for number of epochs
    epochs = questionary.text(
        "Enter number of epochs for training (default: 10):",
        default="10"
    ).ask()
    
    try:
        epochs = int(epochs)
    except ValueError:
        print("Invalid number of epochs. Using default value of 10.")
        epochs = 10
    
    # Ask if user wants to freeze BERT
    freeze_bert = questionary.confirm(
        "Do you want to freeze the BERT base model?",
        default=False
    ).ask()
    
    # Create a directory for cross-validation runs
    cv_base_dir = Path("runs") / f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cv_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Display summary
    print("\nCross-Validation Run Summary:")
    print(f"Number of configurations: {num_configs}")
    print(f"Number of folds: {n_folds}")
    print(f"Number of epochs: {epochs}")
    print(f"BERT frozen: {'Yes' if freeze_bert else 'No'}")
    print(f"Output directory: {cv_base_dir}")
    print("\nSelected configurations:")
    
    for i, run in enumerate(top_runs):
        params_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in run['parameters'].items()])
        metric_value = run['metrics'].get(rank_metric)
        print(f"{i+1}. {run['run_name']} - {rank_metric}: {metric_value:.4f} - {params_str}")
    
    # Ask for confirmation
    proceed = questionary.confirm(
        "Do you want to proceed with these cross-validation runs?",
        default=True
    ).ask()
    
    if not proceed:
        return
    
    # Run cross-validation for each configuration
    for i, run in enumerate(top_runs):
        config_path = run['config_path']
        if not config_path.exists():
            print(f"Config file not found for {run['run_name']}. Skipping.")
            continue
        
        # Create directory for this configuration
        config_dir = cv_base_dir / f"config_{i+1}"
        config_dir.mkdir(exist_ok=True)
        
        # Copy and modify the config file to include n_folds and freeze_bert
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Add cross-validation settings
        if 'data' not in config_data:
            config_data['data'] = {}
        config_data['data']['n_folds'] = n_folds
        
        # Update training epochs
        if 'training' not in config_data:
            config_data['training'] = {}
        config_data['training']['num_epochs'] = epochs
        
        # Add freeze_bert setting
        if 'model' not in config_data:
            config_data['model'] = {}
        config_data['model']['freeze_bert'] = freeze_bert
        
        # Save modified config
        new_config_path = config_dir / "config.yaml"
        with open(new_config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        # Create a params file for reference
        with open(config_dir / "params.json", 'w') as f:
            json.dump({
                "original_run": run['run_name'],
                "parameters": run['parameters'],
                "best_metric": {rank_metric: run['metrics'][rank_metric]},
                "n_folds": n_folds,
                "epochs": epochs,
                "freeze_bert": freeze_bert
            }, f, indent=4)
        
        # Execute command for cross-validation
        cmd = ["python", "hyper_main.py", "--config", str(new_config_path), "--cv", "--cv-dir", str(config_dir)]
        print(f"\nExecuting cross-validation for configuration {i+1}/{len(top_runs)}:")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    print(f"\nCross-validation runs completed. Results saved to {cv_base_dir}")

def get_cv_dirs() -> List[Path]:
    """Get all cross-validation directories."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    return sorted([d for d in runs_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("cv_")], 
                  key=lambda x: x.stat().st_mtime, reverse=True)

def get_incomplete_cv_runs(cv_dir: Path) -> List[Path]:
    """Find configurations in a CV run that did not complete successfully."""
    if not cv_dir.exists():
        return []
    
    # Get all config directories
    config_dirs = [d for d in cv_dir.iterdir() if d.is_dir() and d.name.startswith('config_')]
    incomplete_configs = []
    
    logger.info(f"Checking {len(config_dirs)} configurations in {cv_dir}")
    
    for config_dir in config_dirs:
        # Check if the config has a training.log file with completion message
        log_file = config_dir / "training.log"
        completed = False
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    # Check for both completion messages
                    if "Cross-Validation Results Summary" in log_content:
                        completed = True
                        logger.info(f"Config {config_dir.name} has completion message")
                    else:
                        logger.info(f"Config {config_dir.name} log exists but no completion message found")
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        else:
            logger.info(f"Config {config_dir.name} has no training.log file")
        
        # A config is incomplete if it doesn't have a completion message in the log
        if not completed:
            incomplete_configs.append(config_dir)
            logger.info(f"Config {config_dir.name} marked as incomplete")
    
    logger.info(f"Found {len(incomplete_configs)} incomplete configurations")
    return sorted(incomplete_configs, key=lambda x: int(x.name.split('_')[1]))

def resume_cross_validation():
    """Resume an incomplete cross-validation run."""
    # Get available CV directories
    cv_dirs = get_cv_dirs()
    if not cv_dirs:
        print("No cross-validation directories found.")
        return
    
    # Ask user to select a CV directory
    cv_choices = [d.name for d in cv_dirs]
    cv_dir_name = questionary.select(
        "Select a cross-validation directory:",
        choices=cv_choices
    ).ask()
    
    if not cv_dir_name:
        return
    
    cv_dir = Path("runs") / cv_dir_name
    
    # Get incomplete configurations
    incomplete_configs = get_incomplete_cv_runs(cv_dir)
    if not incomplete_configs:
        print(f"No incomplete configurations found in {cv_dir}.")
        return
    
    # Show number of incomplete configurations and ask for confirmation
    print(f"\nFound {len(incomplete_configs)} incomplete configurations:")
    for config_dir in incomplete_configs:
        # Read params.json for configuration details
        params_path = config_dir / "params.json"
        if params_path.exists():
            try:
                with open(params_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                    print(f"- {config_dir.name}:")
                    print(f"  Original run: {params.get('original_run', 'Unknown')}")
                    print(f"  N folds: {params.get('n_folds', 'Unknown')}")
                    print(f"  Epochs: {params.get('epochs', 'Unknown')}")
                    print(f"  BERT frozen: {'Yes' if params.get('freeze_bert', False) else 'No'}")
                    
                    # Check if there's a training.log file
                    log_file = config_dir / "training.log"
                    if log_file.exists():
                        print(f"  Has training.log: Yes")
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                log_content = f.read()
                                if "Cross-validation completed" in log_content:
                                    print(f"  Has completion message: Yes")
                                else:
                                    print(f"  Has completion message: No")
                        except Exception as e:
                            print(f"  Error reading log file: {e}")
                    else:
                        print(f"  Has training.log: No")
            except Exception as e:
                logger.error(f"Error reading params file {params_path}: {e}")
                print(f"- {config_dir.name}: Error reading configuration")
        else:
            print(f"- {config_dir.name}: No configuration details available")
    
    proceed = questionary.confirm(
        f"\nDo you want to continue with these {len(incomplete_configs)} configurations?",
        default=True
    ).ask()
    
    if not proceed:
        return
    
    # Ask if user wants to restart the incomplete configurations
    restart_last = questionary.confirm(
        "Do you want to restart the incomplete configurations completely? If No, only missing folds will be run.",
        default=False
    ).ask()
    
    # Run cross-validation for each incomplete configuration
    for config_dir in incomplete_configs:
        config_path = config_dir / "config.yaml"
        if not config_path.exists():
            print(f"Config file not found for {config_dir.name}. Skipping.")
            continue
        
        # If restart_last is False, try to identify and run only missing folds
        missing_folds = []
        if not restart_last:
            # First, determine the total number of folds from config or params
            n_folds = None
            
            # Try to get n_folds from params.json
            params_path = config_dir / "params.json"
            if params_path.exists():
                try:
                    with open(params_path, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                        n_folds = params.get('n_folds')
                except Exception as e:
                    logger.error(f"Error reading params file {params_path}: {e}")
            
            # If not found in params, try to get from config.yaml
            if n_folds is None:
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if 'data' in config_data and 'n_folds' in config_data['data']:
                            n_folds = config_data['data']['n_folds']
                except Exception as e:
                    logger.error(f"Error reading config file {config_path}: {e}")
            
            # Default to 5 if we couldn't determine the number of folds
            if n_folds is None:
                logger.warning(f"Could not determine number of folds for {config_dir.name}. Using default of 5.")
                n_folds = 5
            
            # Check the training log for completed folds
            log_file = config_dir / "training.log"
            completed_folds = set()
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                        # Look for fold completion messages in the log
                        for fold in range(n_folds):
                            # Check for the exact message pattern in the log
                            fold_completion_msg = f"Completed fold {fold + 1}/{n_folds}"
                            if fold_completion_msg in log_content:
                                logger.info(f"Found completion message for fold {fold} in log")
                                completed_folds.add(fold)
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
            
            # If we didn't find any fold completion messages, fall back to directory-based check
            if not completed_folds:
                logger.info(f"No fold completion messages found in log, checking directories...")
                for fold in range(n_folds):
                    fold_dir = config_dir / f"fold_{fold}"
                    if fold_dir.exists():
                        model_file = fold_dir / "best-model.pt"  # Change to best-model.pt which is what actually gets saved
                        metrics_file = fold_dir / "metrics.json"
                        
                        if model_file.exists() and metrics_file.exists():
                            completed_folds.add(fold)
                            logger.info(f"Fold {fold} is complete based on directory check in {config_dir.name}")
            
            # Determine missing folds
            missing_folds = [fold for fold in range(n_folds) if fold not in completed_folds]
            
            if not missing_folds and not completed_folds:
                # If we didn't find any completed or missing folds, 
                # assume all folds need to be rerun for consistency
                logger.warning(f"No fold information found in {config_dir.name}. Will rerun all folds.")
                missing_folds = list(range(n_folds))
                
            print(f"\nConfiguration {config_dir.name} has {len(missing_folds)} missing folds: {missing_folds}")
            print(f"Completed folds: {sorted(list(completed_folds))}")
        
        # Execute command for cross-validation
        cmd = ["python", "hyper_main.py", "--config", str(config_path), "--cv", "--cv-dir", str(config_dir)]
        
        if restart_last:
            cmd.append("--restart-last")
        elif missing_folds:
            # Specify which folds to run
            cmd.append("--folds")
            cmd.append(",".join(map(str, missing_folds)))
        
        print(f"\nExecuting cross-validation for {config_dir.name}:")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    print(f"\nCross-validation runs completed. Results saved to {cv_dir}")

def main():
    """Main CLI entrypoint."""
    print("Welcome to the Hyperparameter Search CLI")
    print("----------------------------------------")
    
    while True:
        # Main menu
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Start a new hyperparameter search run",
                "Continue an existing hyperparameter search run",
                "View results of completed runs",
                "Rank runs by performance",
                "Start cross-validation with best runs",
                "Resume cross-validation run",
                "Evaluate misclassifications",
                "Launch annotation tool",
                "Exit"
            ]
        ).ask()
        
        if action == "Start a new hyperparameter search run":
            start_new_run()
        elif action == "Continue an existing hyperparameter search run":
            continue_run()
        elif action == "View results of completed runs":
            view_results()
        elif action == "Rank runs by performance":
            rank_runs()
        elif action == "Start cross-validation with best runs":
            start_cross_validation()
        elif action == "Resume cross-validation run":
            resume_cross_validation()
        elif action == "Evaluate misclassifications":
            evaluate_misclassifications()
        elif action == "Launch annotation tool":
            annotation_tool()
        elif action == "Exit":
            print("Goodbye!")
            sys.exit(0)
        
        # Ask if user wants to return to the main menu
        if not questionary.confirm("Return to main menu?", default=True).ask():
            print("Goodbye!")
            break

if __name__ == "__main__":
    main() 