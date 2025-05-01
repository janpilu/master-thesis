#!/usr/bin/env python3

import os
import glob
import pandas as pd
import questionary
from questionary import Style
from rich.console import Console
from rich.table import Table

# Set up a nice style for the questionary prompts
custom_style = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'fg:white bold'),
    ('answer', 'fg:green bold'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green bold'),
    ('separator', 'fg:cyan'),
    ('instruction', 'fg:white'),
    ('text', 'fg:white'),
])

def get_cv_folders(base_path):
    """Get all CV folders in the base path"""
    # Get all directories with cv_ prefix
    cv_folders = [f for f in glob.glob(os.path.join(base_path, "cv_*")) if os.path.isdir(f)]
    return cv_folders

def analyze_best_epochs(scores_file):
    """Analyze the scores.csv file to find the best epoch per fold per config"""
    # Read the scores.csv file
    df = pd.read_csv(scores_file)
    
    # Find the best epoch (lowest val_loss) for each fold of each config
    best_epochs = df.loc[df.groupby(['fold', 'config'])['val_loss'].idxmin()]
    
    # Sort by config and fold for readability
    best_epochs = best_epochs.sort_values(['config', 'fold'])
    
    return best_epochs

def display_summary(best_epochs):
    """Display a summary of the best epochs using rich"""
    console = Console()
    
    table = Table(title="Best Epochs Summary")
    table.add_column("Config", style="cyan")
    table.add_column("Fold", style="cyan")
    table.add_column("Epoch", style="green")
    table.add_column("Val Loss", style="red")
    table.add_column("Val Accuracy", style="green")
    table.add_column("Val F1 Score", style="yellow")
    
    for _, row in best_epochs.iterrows():
        table.add_row(
            str(row['config']),
            str(row['fold']),
            str(row['epoch']),
            f"{row['val_loss']:.4f}",
            f"{row['val_accuracy']:.4f}",
            f"{row['val_f1_score']:.4f}"
        )
    
    console.print(table)

def save_best_epochs(best_epochs, output_path):
    """Save the best epochs to a CSV file"""
    best_epochs.to_csv(output_path, index=False)
    print(f"✅ Saved best epochs to {output_path}")

def main():
    # Base path for runs
    base_path = "/Users/jlangela/dev/master-thesis/runs"
    
    # Get all CV folders
    cv_folders = get_cv_folders(base_path)
    
    if not cv_folders:
        print("❌ No CV folders found in the runs directory.")
        return
    
    # Get folder names only (not full paths) for display
    folder_names = [os.path.basename(folder) for folder in cv_folders]
    
    # Ask user to select a folder
    selected_folder_name = questionary.select(
        "Select a CV folder to analyze:",
        choices=folder_names,
        style=custom_style
    ).ask()
    
    # Find the full path of the selected folder
    selected_folder = next((folder for folder in cv_folders if os.path.basename(folder) == selected_folder_name), None)
    
    if not selected_folder:
        print("❌ Selected folder not found.")
        return
    
    # Path to scores.csv in the selected folder
    scores_file = os.path.join(selected_folder, "scores.csv")
    
    if not os.path.exists(scores_file):
        print(f"❌ scores.csv file not found in {selected_folder}")
        return
    
    # Analyze the scores.csv file
    best_epochs = analyze_best_epochs(scores_file)
    
    # Display summary
    display_summary(best_epochs)
    
    # Ask if user wants to save the results
    save_result = questionary.confirm(
        "Do you want to save the best epochs to a CSV file?",
        style=custom_style
    ).ask()
    
    if save_result:
        output_path = os.path.join(selected_folder, "best_epochs.csv")
        save_best_epochs(best_epochs, output_path)
    
    # Ask if user wants to see stats for the best epochs
    stats_result = questionary.confirm(
        "Do you want to see statistics for the best epochs?",
        style=custom_style
    ).ask()
    
    if stats_result:
        # Calculate mean and std for val_loss, val_accuracy, val_f1_score grouped by config
        stats = best_epochs.groupby('config')[['val_loss', 'val_accuracy', 'val_f1_score']].agg(['mean', 'std'])
        
        console = Console()
        
        stats_table = Table(title="Statistics by Configuration")
        stats_table.add_column("Config", style="cyan")
        stats_table.add_column("Mean Val Loss", style="red")
        stats_table.add_column("Std Val Loss", style="red")
        stats_table.add_column("Mean Val Accuracy", style="green")
        stats_table.add_column("Std Val Accuracy", style="green")
        stats_table.add_column("Mean Val F1 Score", style="yellow")
        stats_table.add_column("Std Val F1 Score", style="yellow")
        
        for config, row in stats.iterrows():
            stats_table.add_row(
                str(config),
                f"{row[('val_loss', 'mean')]:.4f}",
                f"{row[('val_loss', 'std')]:.4f}",
                f"{row[('val_accuracy', 'mean')]:.4f}",
                f"{row[('val_accuracy', 'std')]:.4f}",
                f"{row[('val_f1_score', 'mean')]:.4f}",
                f"{row[('val_f1_score', 'std')]:.4f}"
            )
        
        console.print(stats_table)
        
        # Ask if user wants to save the statistics
        stats_save = questionary.confirm(
            "Do you want to save these statistics to a CSV file?",
            style=custom_style
        ).ask()
        
        if stats_save:
            stats_path = os.path.join(selected_folder, "best_epochs_stats.csv")
            stats.to_csv(stats_path)
            print(f"✅ Saved statistics to {stats_path}")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import questionary
        import pandas as pd
        import rich
    except ImportError:
        print("Please install the required packages:")
        print("pip install questionary pandas rich")
        exit(1)
        
    main() 