#!/usr/bin/env python
"""
Script to evaluate model misclassifications.
Loads a model, runs on validation set, picks random misclassifications for manual review,
and exports results to CSV.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import random
import csv
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.data.toxigen import ToxiGenDataModule
from src.models.factory import ModelFactory
from src.utils.config import Config

def get_device():
    """Get the best available device for evaluation."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model_and_config(model_path: str, config_path: str):
    """Load a trained model and its configuration.
    
    Args:
        model_path: Path to the saved model checkpoint
        config_path: Path to the config file used for training
        
    Returns:
        tuple: (model, config)
    """
    config = Config(config_path=config_path)
    model = ModelFactory.create_model(config)
    
    # Load the saved model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def find_best_checkpoint(run_dir: str) -> Tuple[str, str]:
    """Find the best checkpoint and config file in a run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        tuple: (best_checkpoint_path, config_path)
    """
    run_path = Path(run_dir)
    config_path = run_path / "config.yaml"
    checkpoint_dir = run_path / "checkpoints"
    
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
        
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in run directory: {config_path}")
        
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoint_dir}")
    
    # Look for best checkpoint first in the run root directory (best-model.pt)
    best_model_root = run_path / "best-model.pt"
    if best_model_root.exists():
        print(f"Found best model in run root: {best_model_root}")
        return str(best_model_root), str(config_path)
    
    # Then look for best checkpoint in checkpoints directory
    best_checkpoint = checkpoint_dir / "best-model.pt"
    if best_checkpoint.exists():
        print(f"Found best checkpoint: {best_checkpoint}")
        return str(best_checkpoint), str(config_path)
    
    # Legacy format: best_model.pt
    legacy_best = checkpoint_dir / "best_model.pt"
    if legacy_best.exists():
        print(f"Found legacy best checkpoint: {legacy_best}")
        return str(legacy_best), str(config_path)
    
    # If no best-model.pt, look for any checkpoint with "best" in the name
    best_candidates = list(checkpoint_dir.glob("*best*.pt"))
    if best_candidates:
        # Sort by modification time (newest first)
        best_checkpoint = sorted(best_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        print(f"Found best checkpoint: {best_checkpoint}")
        return str(best_checkpoint), str(config_path)
    
    # If still no best checkpoint, take the latest one
    all_checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    latest_checkpoint = sorted(all_checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"No best checkpoint found. Using latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint), str(config_path)

def find_latest_run() -> str:
    """Find the most recent run directory.
    
    Returns:
        Path to the most recent run directory
    """
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise FileNotFoundError("Runs directory not found")
    
    # Look for both date format (YYYY-MM-DD) and legacy format (run_timestamp)
    date_format_runs = [d for d in runs_dir.iterdir() if d.is_dir() and 
                      (len(d.name) == 10 and d.name.count('-') == 2)]  # YYYY-MM-DD format
    legacy_runs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    all_runs = date_format_runs + legacy_runs
    if not all_runs:
        raise FileNotFoundError("No run directories found in runs directory")
    
    latest_run = sorted(all_runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"Using most recent run: {latest_run}")
    return str(latest_run)

def get_misclassifications(model, val_loader, device, n_samples=20, sort_by_confidence=False) -> List[Dict]:
    """Evaluate model and find misclassifications.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run evaluation on
        n_samples: Number of misclassifications to return
        sort_by_confidence: Whether to sort by confidence (highest first)
        
    Returns:
        List of dictionaries containing misclassified examples
    """
    model.eval()
    misclassifications = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating model")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get model predictions
            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
            predictions = (probabilities > 0.5).float() if outputs.shape[1] == 1 else torch.argmax(probabilities, dim=1)
            
            # Find misclassifications in this batch
            misclassified_indices = (predictions != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_indices:
                # Try to get the text in different ways (adapt to your dataset structure)
                text = None
                
                # Method 1: If "text" is directly in the batch
                if "text" in batch and idx < len(batch["text"]):
                    text = batch["text"][idx]
                
                # Method 2: Calculate dataset index and look it up
                dataset_idx = batch_idx * val_loader.batch_size + idx.item()
                if dataset_idx < len(val_loader.dataset):
                    if hasattr(val_loader.dataset, "dataset"):
                        # For dataset wrapping another dataset
                        raw_item = val_loader.dataset.dataset[dataset_idx]
                        if isinstance(raw_item, dict) and "text" in raw_item:
                            text = raw_item["text"]
                    
                    # Try getting item directly (this might return a tuple or dict)
                    if text is None:
                        item = val_loader.dataset[dataset_idx]
                        if isinstance(item, dict) and "text" in item:
                            text = item["text"]
                
                # If we still haven't found the text, log the issue but continue
                if text is None:
                    text = f"[Could not retrieve text for example {dataset_idx}]"
                    print(f"Warning: Could not retrieve text for misclassified example at index {dataset_idx}")
                
                # Get prediction and true label
                model_pred = predictions[idx].item()
                true_label = labels[idx].item()
                
                # Get probability/confidence
                if outputs.shape[1] == 1:  # Binary classification
                    confidence = probabilities[idx].item()
                    # If prediction is 0, confidence is 1 - probability
                    if model_pred == 0:
                        confidence = 1 - confidence
                else:  # Multi-class classification
                    confidence = probabilities[idx][model_pred].item()
                
                misclassifications.append({
                    "text": text,
                    "model_prediction": model_pred,
                    "true_label": true_label,
                    "confidence": confidence,
                    "idx": dataset_idx
                })
    
    # If requested, sort by confidence (highest first)
    if sort_by_confidence and misclassifications:
        misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Randomly sample from misclassifications if we have more than requested
    if len(misclassifications) > n_samples:
        if sort_by_confidence:
            # If sorted, just take top N
            misclassifications = misclassifications[:n_samples]
        else:
            # Otherwise random sample
            misclassifications = random.sample(misclassifications, n_samples)
    
    return misclassifications

def get_user_annotation(example: Dict, output_format: Dict = None) -> Dict:
    """Present a single example to the user and get their annotation.
    
    Args:
        example: Dictionary containing text and predictions
        output_format: Dictionary mapping from label indices to readable labels
        
    Returns:
        Updated example with user annotation
    """
    print("\n" + "="*80)
    print(f"TEXT: {example['text']}")
    print("-"*80)
    
    # Present options without revealing truth or model prediction
    if output_format:
        print("\nPossible labels:")
        for key, value in output_format.items():
            print(f"{key}: {value}")
    else:
        print("\nProvide your classification as a number.")
    
    # Get user input
    valid_input = False
    while not valid_input:
        user_input = input("\nYour classification (or 'skip' to skip): ")
        
        if user_input.lower() == 'skip':
            example['user_annotation'] = 'skipped'
            valid_input = True
        elif output_format and user_input in output_format:
            example['user_annotation'] = user_input
            valid_input = True
        elif user_input.isdigit() or (user_input.startswith('-') and user_input[1:].isdigit()):
            example['user_annotation'] = int(user_input)
            valid_input = True
        else:
            print("Invalid input. Please try again.")
    
    # Now that we have the user's annotation, reveal the truth and model prediction
    print("\n" + "-"*80)
    print("REVEALED INFORMATION:")
    
    # Display labels based on output format
    if output_format:
        true_label_text = output_format.get(str(example['true_label']), str(example['true_label']))
        model_pred_text = output_format.get(str(example['model_prediction']), str(example['model_prediction']))
        print(f"TRUE LABEL: {true_label_text} ({example['true_label']})")
        print(f"MODEL PREDICTION: {model_pred_text} ({example['model_prediction']}) with {example['confidence']:.2%} confidence")
        
        # Show user's choice in the same format
        if example['user_annotation'] == 'skipped':
            user_text = "SKIPPED"
        else:
            user_text = output_format.get(str(example['user_annotation']), str(example['user_annotation']))
        print(f"YOUR ANNOTATION: {user_text}")
    else:
        print(f"TRUE LABEL: {example['true_label']}")
        print(f"MODEL PREDICTION: {example['model_prediction']} with {example['confidence']:.2%} confidence")
        print(f"YOUR ANNOTATION: {example['user_annotation']}")
    
    return example

def export_to_csv(annotations: List[Dict], output_file: str, output_format: Dict = None):
    """Export annotations to a CSV file.
    
    Args:
        annotations: List of annotated examples
        output_file: Path to save the CSV file
        output_format: Dictionary mapping from label indices to readable labels
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'true_label', 'model_prediction', 'user_annotation', 'confidence']
        
        # Add human-readable fields if output format is provided
        if output_format:
            fieldnames.extend(['true_label_text', 'model_prediction_text', 'user_annotation_text'])
            
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in annotations:
            row = {
                'text': item['text'],
                'true_label': item['true_label'],
                'model_prediction': item['model_prediction'],
                'user_annotation': item['user_annotation'],
                'confidence': f"{item['confidence']:.4f}"
            }
            
            # Add human-readable labels if output format is provided
            if output_format:
                row['true_label_text'] = output_format.get(str(item['true_label']), 'Unknown')
                row['model_prediction_text'] = output_format.get(str(item['model_prediction']), 'Unknown')
                
                # Handle skipped annotations
                if item['user_annotation'] == 'skipped':
                    row['user_annotation_text'] = 'Skipped'
                else:
                    row['user_annotation_text'] = output_format.get(str(item['user_annotation']), 'Unknown')
                    
            writer.writerow(row)
    
    print(f"\nAnnotations saved to {output_file}")

def generate_confusion_matrix(all_misclassifications, output_format=None):
    """Generate and display a confusion matrix for all misclassifications.
    
    Args:
        all_misclassifications: List of dictionaries containing misclassified examples
        output_format: Dictionary mapping from label indices to readable labels
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Warning: Could not generate confusion matrix. Missing dependencies.")
        return
    
    # Extract true labels and model predictions
    y_true = [example['true_label'] for example in all_misclassifications]
    y_pred = [example['model_prediction'] for example in all_misclassifications]
    
    # Get unique labels, ensuring they're sorted
    if output_format:
        all_labels = sorted([int(k) for k in output_format.keys() if k.isdigit()])
    else:
        all_labels = sorted(set(y_true + y_pred))
    
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Create a readable labels dictionary for display
    if output_format:
        label_names = [output_format.get(str(label), str(label)) for label in all_labels]
    else:
        label_names = [str(label) for label in all_labels]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of Misclassifications')
    
    # Save and show the confusion matrix
    plt.tight_layout()
    plt.savefig('misclassification_confusion_matrix.png')
    print("\nConfusion matrix saved to 'misclassification_confusion_matrix.png'")
    
    # Also print some statistics
    total_errors = len(all_misclassifications)
    print(f"\nTotal misclassifications: {total_errors}")
    
    # Count errors per class
    errors_per_class = {}
    for example in all_misclassifications:
        true_label = example['true_label']
        if true_label not in errors_per_class:
            errors_per_class[true_label] = 0
        errors_per_class[true_label] += 1
    
    print("\nErrors per class:")
    for label, count in sorted(errors_per_class.items()):
        if output_format:
            label_name = output_format.get(str(label), str(label))
            print(f"  {label_name} ({label}): {count} ({count/total_errors:.1%})")
        else:
            print(f"  {label}: {count} ({count/total_errors:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model misclassifications and annotate them")
    parser.add_argument("--model", type=str, help="Path to the saved model checkpoint")
    parser.add_argument("--config", type=str, help="Path to the config file used for training")
    parser.add_argument("--run-dir", type=str, help="Path to a specific run directory (will use best checkpoint)")
    parser.add_argument("--latest-run", action="store_true", help="Use the most recent run")
    parser.add_argument("--output", type=str, default="annotations.csv", help="Output CSV file path")
    parser.add_argument("--samples", type=int, default=20, help="Number of misclassifications to review")
    parser.add_argument("--sort-by-confidence", action="store_true", help="Sort misclassifications by confidence (highest first)")
    parser.add_argument("--all-misclassifications", action="store_true", help="Collect all misclassifications (potentially memory intensive)")
    parser.add_argument("--confusion-matrix", action="store_true", help="Generate a confusion matrix of misclassifications")
    
    args = parser.parse_args()
    
    # Determine model and config paths based on arguments
    model_path = None
    config_path = None
    
    if args.run_dir:
        # Use specified run directory
        model_path, config_path = find_best_checkpoint(args.run_dir)
    elif args.latest_run:
        # Find the most recent run
        latest_run = find_latest_run()
        model_path, config_path = find_best_checkpoint(latest_run)
    elif args.model and args.config:
        # Use explicitly provided paths
        model_path = args.model
        config_path = args.config
    else:
        parser.error("Either --run-dir, --latest-run, or both --model and --config must be provided")
    
    # Load model and config
    model, config = load_model_and_config(model_path, config_path)
    device = get_device()
    model = model.to(device)
    
    # Load data
    data_module = ToxiGenDataModule(config)
    data_module.setup()
    dataloaders = data_module.get_dataloaders()
    val_loader = dataloaders["test"]  # Using test set as validation
    
    # Get label format if available in config
    output_format = None
    if hasattr(config, 'label_mapping') and config.label_mapping:
        output_format = config.label_mapping
    else:
        # For binary classification, create a simple mapping
        output_format = {'0': 'Not Toxic', '1': 'Toxic'}
    
    # Set the number of samples to a large number if we want all misclassifications
    n_samples = 10000 if args.all_misclassifications else args.samples
    
    # Find misclassifications
    print(f"Finding{' all' if args.all_misclassifications else ''} misclassifications..." + 
          (" (sorted by confidence)" if args.sort_by_confidence else ""))
    misclassifications = get_misclassifications(
        model, val_loader, device, n_samples, sort_by_confidence=args.sort_by_confidence
    )
    
    # No misclassifications found
    if not misclassifications:
        print("No misclassifications found! The model might be perfect... or there's an issue.")
        return
    
    print(f"Found {len(misclassifications)} misclassifications.")
    
    # Generate confusion matrix if requested
    if args.confusion_matrix:
        generate_confusion_matrix(misclassifications, output_format)
    
    # Take a sample for annotation if we found all misclassifications
    samples_to_annotate = misclassifications
    if args.all_misclassifications and len(misclassifications) > args.samples:
        if args.sort_by_confidence:
            samples_to_annotate = misclassifications[:args.samples]
        else:
            samples_to_annotate = random.sample(misclassifications, args.samples)
        print(f"Selecting {args.samples} samples for annotation.")
    
    # Get user annotations
    annotations = []
    for example in samples_to_annotate:
        annotated_example = get_user_annotation(example, output_format)
        annotations.append(annotated_example)
    
    # Export to CSV
    export_to_csv(annotations, args.output, output_format)

if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main() 