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