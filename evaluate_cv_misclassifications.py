#!/usr/bin/env python
"""Script to evaluate misclassifications from a cross-validation model."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import questionary
import scipy.stats as stats
import seaborn as sns
import torch
from tqdm import tqdm

from src.data.toxigen import ToxiGenDataModule
from src.models.factory import ModelFactory
from src.training.metrics import accuracy_metric
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device for evaluation."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_cv_dirs() -> List[Path]:
    """Get all cross-validation directories."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []

    return sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("cv_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )


def get_fold_dirs(cv_dir: Path) -> List[Path]:
    """Get all fold directories within a config directory."""
    fold_dirs = []
    for config_dir in cv_dir.iterdir():
        if config_dir.is_dir() and config_dir.name.startswith("config_"):
            for fold_dir in config_dir.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                    fold_dirs.append(fold_dir)
    return fold_dirs


def format_fold_choice(fold_dir: Path) -> str:
    """Format a fold directory as a choice for questionary."""
    # Read the metrics file to get the fold's performance
    metrics_file = fold_dir / "metrics.json"
    metrics_str = ""

    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        except Exception as e:
            logger.error(f"Error reading metrics file {metrics_file}: {e}")

    config_name = fold_dir.parent.name
    fold_name = fold_dir.name
    return f"{config_name}/{fold_name} ({metrics_str})"


def load_model_and_config(fold_dir: Path) -> Tuple[torch.nn.Module, Config, int]:
    """Load model and config from a fold directory."""
    model_path = fold_dir / "best-model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load fold index from directory name - extract number after 'fold_'
    fold_name = fold_dir.name
    fold_idx = int(fold_name.split("_")[1])
    print(f"Loaded fold directory: {fold_name}, extracted fold index: {fold_idx}")

    # Load config
    config_path = fold_dir.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = Config(config_path=str(config_path))

    # Create model
    model = ModelFactory.create_model(config)

    # Load model weights
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Check if the checkpoint contains the fold index and use it if available
    if "fold_idx" in checkpoint:
        checkpoint_fold_idx = checkpoint["fold_idx"]
        print(f"Checkpoint contains fold_idx: {checkpoint_fold_idx}")
        if checkpoint_fold_idx is not None and checkpoint_fold_idx != fold_idx - 1:
            print(
                f"WARNING: Checkpoint fold index ({checkpoint_fold_idx}) doesn't match directory fold index minus 1 ({fold_idx - 1})"
            )

    return model, config, fold_idx


def evaluate_misclassifications(
    model: torch.nn.Module, config: Config, fold_idx: int
) -> List[Dict]:
    """Evaluate the model and find misclassifications."""
    device = get_device()

    # Set random seed for reproducibility - essential to get the same fold split
    random_seed = config.training_config.get("random_seed", 42)
    print(f"Using random seed: {random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(random_seed)

    # Set up data module with the same fold index
    data_module = ToxiGenDataModule(config)

    # Important: fold_idx in directory names is 1-based (fold_1, fold_2, etc.),
    # but in the code it's 0-based (0, 1, 2, etc.)
    # Convert to 0-based index for the data module
    zero_based_fold_idx = fold_idx - 1 if fold_idx > 0 else fold_idx
    print(
        f"Directory fold index: {fold_idx}, Zero-based fold index for data module: {zero_based_fold_idx}"
    )

    # Set the current fold index to ensure we get the correct test split
    if hasattr(data_module, "current_fold"):
        data_module.current_fold = zero_based_fold_idx

    data_module.setup()

    # Get the test dataloader - use get_dataloaders() and extract the test loader
    dataloaders = data_module.get_dataloaders()
    test_loader = dataloaders["test"]
    print(f"Test loader size: {len(test_loader)} batches")

    # Run evaluation
    misclassifications = []
    model.eval()

    # For accuracy calculation - using the official metrics
    all_outputs = []
    all_labels = []

    # For detailed analysis
    all_predictions = []
    count_by_class = {0: 0, 1: 0}
    correct_by_class = {0: 0, 1: 0}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ["labels", "text", "metadata"]
            }
            labels = (
                batch["labels"].to(device)
                if isinstance(batch["labels"], torch.Tensor)
                else batch["labels"]
            )
            text = batch["text"]

            outputs = model(**inputs)

            # Get prediction probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            # Store outputs and labels for metric calculation
            all_outputs.append(outputs)
            all_labels.append(labels)
            all_predictions.append(preds)

            # Count by class and correct predictions
            for i, (pred, label) in enumerate(zip(preds, labels)):
                pred_item = pred.item()
                label_item = label.item()
                count_by_class[label_item] = count_by_class.get(label_item, 0) + 1

                # If prediction doesn't match label, add to misclassifications
                if pred_item != label_item:
                    # Extract the prediction probabilities for this sample
                    probs = probabilities[i].cpu().tolist()

                    # Get original score if available
                    original_score = None
                    if "metadata" in batch:
                        if (
                            isinstance(batch["metadata"], dict)
                            and "toxicity_human" in batch["metadata"]
                        ):
                            # If metadata is a dictionary with toxicity_human key containing lists/tensors
                            if isinstance(
                                batch["metadata"]["toxicity_human"],
                                (list, torch.Tensor),
                            ):
                                original_score = batch["metadata"]["toxicity_human"][i]
                            else:
                                original_score = batch["metadata"]["toxicity_human"]
                        elif i < len(batch["metadata"]) and isinstance(
                            batch["metadata"][i], dict
                        ):
                            # If metadata is a list of dictionaries
                            original_score = batch["metadata"][i].get("toxicity_human")

                    misclassifications.append(
                        {
                            "text": text[i],
                            "predicted": pred_item,
                            "predicted_probs": probs,  # Store softmax probabilities
                            "actual": label_item,
                            "original_score": original_score,  # May be None if not available
                        }
                    )
                elif pred_item == label_item:
                    correct_by_class[label_item] = (
                        correct_by_class.get(label_item, 0) + 1
                    )

    # Calculate and print accuracy using the same method as in training
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    # Convert to numpy for easier metric calculation
    y_true = all_labels.cpu().numpy()
    y_pred = all_predictions.cpu().numpy()

    # Calculate metrics
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # True Positive Rate (Sensitivity, Recall)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    # True Negative Rate (Specificity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 Score
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

    # Matthews Correlation Coefficient
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    # Overall accuracy
    accuracy = accuracy_metric(all_outputs, all_labels) * 100

    print("\nDetailed Validation Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"True Positive Rate (Sensitivity/Recall): {tpr:.4f}")
    print(f"True Negative Rate (Specificity): {tnr:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")

    # Add metrics to each misclassification for later analysis
    for item in misclassifications:
        item["metrics"] = {
            "accuracy": float(accuracy),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "precision": float(precision),
            "f1_score": float(f1),
            "mcc": float(mcc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    print(f"Misclassifications: {len(misclassifications)}")

    return misclassifications


def prompt_for_corrections(misclassifications: List[Dict]) -> List[Dict]:
    """Prompt the user for corrections to misclassifications."""
    corrected_items = []

    print(f"\nFound {len(misclassifications)} misclassifications.")
    if not misclassifications:
        return corrected_items

    review_all = questionary.confirm(
        "Do you want to review all misclassifications?", default=True
    ).ask()

    if not review_all:
        max_count = questionary.text(
            "How many misclassifications do you want to review?", default="10"
        ).ask()
        try:
            max_count = int(max_count)
            misclassifications = misclassifications[:max_count]
        except ValueError:
            print("Invalid number, showing 10 misclassifications")
            misclassifications = misclassifications[:10]

    # Get label names from config
    label_names = ["benign", "toxic"]  # Default if not found

    print("\nReview misclassifications:")
    for i, item in enumerate(misclassifications):
        print(f"\n{'-' * 80}")
        print(f"Example {i + 1}/{len(misclassifications)}")
        print(f"Text: {item['text']}")

        # Don't show predicted and actual labels, as requested by the user

        # Ask for correction
        choices = [f"{i}: {name}" for i, name in enumerate(label_names)]
        choices.append("Skip this example")

        correction = questionary.select(
            "What should be the correct label?", choices=choices
        ).ask()

        if correction != "Skip this example":
            corrected_label = int(correction.split(":")[0])

            # Store all the information in the corrected item
            corrected_items.append(
                {
                    "text": item["text"],
                    "predicted": item["predicted"],
                    "predicted_probs": item["predicted_probs"],
                    "actual": item["actual"],
                    "original_score": item.get("original_score"),
                    "corrected": corrected_label,
                }
            )

    return corrected_items


def save_results(corrected_items: List[Dict], output_dir: Path):
    """Save the corrected misclassifications to a file."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"corrected_misclassifications_{timestamp}.json"

    # Format the output data with more detailed information
    formatted_items = []
    for item in corrected_items:
        # Handle both old and new data structures
        if "model_prediction" in item:
            # New structure - already formatted correctly
            formatted_item = item
        else:
            # Old structure - needs formatting
            # Prepare original_score - ensure it's a number or null, not None
            original_score = item.get("original_score")
            if original_score is not None:
                # Convert to float if it's a tensor or numpy array
                if hasattr(original_score, "item"):
                    original_score = original_score.item()
                elif hasattr(original_score, "tolist"):
                    original_score = original_score.tolist()
                # Ensure it's a float
                try:
                    original_score = float(original_score)
                except (ValueError, TypeError):
                    original_score = None

            formatted_item = {
                "text": item["text"],
                "model_prediction": {
                    "class": item["predicted"],
                    "probabilities": item["predicted_probs"],
                },
                "ground_truth": {
                    "binary_class": item["actual"],
                    "original_score": original_score,
                },
                "human_correction": item.get("corrected"),
                "metrics": item.get("metrics", {}),
            }
        formatted_items.append(formatted_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_items, f, indent=4)

    print(f"\nCorrected misclassifications saved to {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model misclassifications from a cross-validation run"
    )
    parser.add_argument("--model", type=str, help="Path to model file (optional)")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--fold", type=int, help="Fold index (optional)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="misclassifications",
        help="Directory to save misclassification results",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="If set, skip manual annotation and output all misclassifications with human_correction as null.",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # If model and config paths are provided, use them directly
    if args.model and args.config and args.fold is not None:
        model_path = Path(args.model)
        config_path = Path(args.config)
        fold_idx = args.fold

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return

        config = Config(config_path=str(config_path))
        model = ModelFactory.create_model(config)

        device = get_device()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    else:
        # Interactive selection of CV run and fold
        cv_dirs = get_cv_dirs()
        if not cv_dirs:
            logger.error("No cross-validation directories found.")
            return

        # Ask user to select a CV directory
        cv_choices = [d.name for d in cv_dirs]
        cv_dir_name = questionary.select(
            "Select a cross-validation directory:", choices=cv_choices
        ).ask()

        if not cv_dir_name:
            return

        cv_dir = Path("runs") / cv_dir_name

        # Get all fold directories
        fold_dirs = get_fold_dirs(cv_dir)
        if not fold_dirs:
            logger.error(f"No fold directories found in {cv_dir}.")
            return

        # Filter folds with best-model.pt
        valid_fold_dirs = [d for d in fold_dirs if (d / "best-model.pt").exists()]
        if not valid_fold_dirs:
            logger.error(f"No folds with best-model.pt found in {cv_dir}.")
            return

        # Format fold choices
        fold_choices = [format_fold_choice(fold_dir) for fold_dir in valid_fold_dirs]

        # Ask user to select a fold
        fold_choice = questionary.select(
            "Select a fold to evaluate:", choices=fold_choices
        ).ask()

        if not fold_choice:
            return

        # Extract fold directory from choice
        selected_config_fold = fold_choice.split(" ")[0]
        config_name, fold_name = selected_config_fold.split("/")
        selected_fold_dir = cv_dir / config_name / fold_name

        # Extract the fold index from the directory name (e.g., fold_3 -> 3)
        fold_idx = int(fold_name.split("_")[1])
        print(f"Selected fold: {fold_name}, Fold index: {fold_idx}")

        # Load model and config
        try:
            model, config, _ = load_model_and_config(selected_fold_dir)
        except Exception as e:
            logger.error(f"Error loading model and config: {e}")
            return

    # Evaluate misclassifications
    try:
        misclassifications = evaluate_misclassifications(model, config, fold_idx)

        # Create DataFrame for plotting
        df = pd.DataFrame(
            [
                {
                    "normalized_score": float(item.get("original_score", 0))
                    if item.get("original_score") is not None
                    else None,  # Ensure float type
                    "toxic_prob": float(
                        item["predicted_probs"][1]
                    ),  # Ensure float type
                    "binary_class": int(item["actual"]),  # Ensure int type
                    "predicted_class": int(item["predicted"]),  # Ensure int type
                    "metrics": item.get("metrics", {}),
                    "text": item["text"][:50] + "..."
                    if len(item["text"]) > 50
                    else item["text"],
                }
                for item in misclassifications
            ]
        )

        # Normalize scores if they exist (from 1-5 scale to 0-1 scale)
        if "normalized_score" in df.columns:
            df["normalized_score"] = df["normalized_score"].apply(
                lambda x: (x - 1) / 4.0 if pd.notnull(x) else None
            )
            # Convert to float type explicitly
            df["normalized_score"] = pd.to_numeric(
                df["normalized_score"], errors="coerce"
            )
            df["toxic_prob"] = pd.to_numeric(df["toxic_prob"], errors="coerce")

        # Calculate correlation stats only if we have normalized scores
        if (
            not df.empty
            and "normalized_score" in df.columns
            and df["normalized_score"].notna().any()
        ):
            valid_mask = df["normalized_score"].notna()
            correlation, p_value = stats.pearsonr(
                df.loc[valid_mask, "normalized_score"].values.astype(float),
                df.loc[valid_mask, "toxic_prob"].values.astype(float),
            )
            spearman_corr, spearman_p = stats.spearmanr(
                df.loc[valid_mask, "normalized_score"].values.astype(float),
                df.loc[valid_mask, "toxic_prob"].values.astype(float),
            )

            # Calculate confidences only for rows with normalized scores
            df.loc[valid_mask, "model_confidence"] = (
                abs(df.loc[valid_mask, "toxic_prob"] - 0.5) * 2
            )
            df.loc[valid_mask, "annotator_confidence"] = (
                abs(df.loc[valid_mask, "normalized_score"] - 0.5) * 2
            )

            confidence_corr, confidence_p = stats.pearsonr(
                df.loc[valid_mask, "model_confidence"].values.astype(float),
                df.loc[valid_mask, "annotator_confidence"].values.astype(float),
            )

            stats_dict = {
                "pearson_correlation": float(correlation),  # Ensure float type
                "pearson_p_value": float(p_value),  # Ensure float type
                "spearman_correlation": float(spearman_corr),  # Ensure float type
                "spearman_p_value": float(spearman_p),  # Ensure float type
                "confidence_correlation": float(confidence_corr),  # Ensure float type
                "confidence_p_value": float(confidence_p),  # Ensure float type
            }

            # Generate plots
            output_dir = Path(args.output_dir)
            plot_results(df, stats_dict, output_dir)

        if args.no_annotate:
            # Output all misclassifications with human_correction as null
            corrected_items = []
            for item in misclassifications:
                corrected_items.append(
                    {
                        "text": item["text"],
                        "model_prediction": {
                            "class": item["predicted"],
                            "probabilities": item["predicted_probs"],
                        },
                        "ground_truth": {
                            "binary_class": item["actual"],
                            "original_score": item.get("original_score"),
                        },
                        "human_correction": None,
                        "metrics": item.get("metrics", {}),
                    }
                )
        else:
            corrected_items = prompt_for_corrections(misclassifications)

        if corrected_items:
            output_dir = Path(args.output_dir)
            save_results(corrected_items, output_dir)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


def plot_results(df, stats, output_dir):
    """Generate plots to visualize the analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Scatter plot of normalized scores vs. toxic probabilities
    plt.figure(figsize=(10, 10))
    plt.scatter(df["normalized_score"], df["toxic_prob"], alpha=0.6)
    plt.plot([0, 1], [0, 1], "r--", label="Perfect correlation")
    plt.xlabel("Normalized Annotator Score (0=benign, 1=toxic)")
    plt.ylabel("Model Toxic Probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(
        f"Annotator Score vs. Model Probability\nPearson r={stats['pearson_correlation']:.3f}, p={stats['pearson_p_value']:.3f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "score_vs_probability.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 2. Scatter plot of confidence levels
    plt.figure(figsize=(10, 10))
    plt.scatter(df["annotator_confidence"], df["model_confidence"], alpha=0.6)
    plt.plot([0, 1], [0, 1], "r--", label="Perfect correlation")
    plt.xlabel("Annotator Confidence (0=uncertain, 1=certain)")
    plt.ylabel("Model Confidence (0=uncertain, 1=certain)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(
        f"Annotator Confidence vs. Model Confidence\nPearson r={stats['confidence_correlation']:.3f}, p={stats['confidence_p_value']:.3f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "confidence_correlation.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 3. Histogram of normalized scores by true class
    plt.figure(figsize=(10, 6))
    df_benign = df[df["binary_class"] == 0]
    df_toxic = df[df["binary_class"] == 1]

    plt.hist(df_benign["normalized_score"], alpha=0.5, bins=20, label="Benign class")
    plt.hist(df_toxic["normalized_score"], alpha=0.5, bins=20, label="Toxic class")
    plt.xlabel("Normalized Annotator Score (0=benign, 1=toxic)")
    plt.ylabel("Count")
    plt.title("Distribution of Annotator Scores by True Class")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "score_distribution_by_class.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 4. Box plot of model probabilities vs. binary classes
    plt.figure(figsize=(10, 6))
    df_plot = pd.DataFrame(
        {"True Benign": df_benign["toxic_prob"], "True Toxic": df_toxic["toxic_prob"]}
    )
    df_plot.boxplot()
    plt.ylabel("Model Toxic Probability")
    plt.title("Model Probability Distribution by True Class")
    plt.savefig(
        os.path.join(output_dir, "probability_boxplot_by_class.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 5. Heat map of uncertainty regions
    plt.figure(figsize=(10, 8))
    heatmap, xedges, yedges = np.histogram2d(
        df["normalized_score"], df["toxic_prob"], bins=10, range=[[0, 1], [0, 1]]
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Count")
    plt.xlabel("Normalized Annotator Score (0=benign, 1=toxic)")
    plt.ylabel("Model Toxic Probability")
    plt.title("Density of Misclassifications")
    plt.savefig(
        os.path.join(output_dir, "misclassification_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 6. Confusion Matrix Plot
    if "metrics" in df.iloc[0] and "confusion_matrix" in df.iloc[0]["metrics"]:
        plt.figure(figsize=(8, 6))
        cm = np.array(
            [
                [
                    df.iloc[0]["metrics"]["confusion_matrix"]["tn"],
                    df.iloc[0]["metrics"]["confusion_matrix"]["fp"],
                ],
                [
                    df.iloc[0]["metrics"]["confusion_matrix"]["fn"],
                    df.iloc[0]["metrics"]["confusion_matrix"]["tp"],
                ],
            ]
        )

        # Calculate percentages for annotations
        cm_sum = np.sum(cm)
        cm_percentages = cm / cm_sum * 100

        # Create annotations with both count and percentage
        annot = np.array(
            [
                [
                    f"{cm[0, 0]}\n({cm_percentages[0, 0]:.1f}%)",
                    f"{cm[0, 1]}\n({cm_percentages[0, 1]:.1f}%)",
                ],
                [
                    f"{cm[1, 0]}\n({cm_percentages[1, 0]:.1f}%)",
                    f"{cm[1, 1]}\n({cm_percentages[1, 1]:.1f}%)",
                ],
            ]
        )

        # Create heatmap
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="RdYlBu_r",
            xticklabels=["Predicted Benign", "Predicted Toxic"],
            yticklabels=["True Benign", "True Toxic"],
            cbar=True,
            cbar_kws={"label": "Count"},
        )

        # Add metrics as text
        plt.text(
            -0.4,
            -0.3,
            f"TPR: {df.iloc[0]['metrics']['tpr']:.3f}\n"
            + f"TNR: {df.iloc[0]['metrics']['tnr']:.3f}\n"
            + f"Precision: {df.iloc[0]['metrics']['precision']:.3f}\n"
            + f"F1: {df.iloc[0]['metrics']['f1_score']:.3f}\n"
            + f"MCC: {df.iloc[0]['metrics']['mcc']:.3f}",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "confusion_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.close("all")


if __name__ == "__main__":
    main()
