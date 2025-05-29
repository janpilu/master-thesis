#!/usr/bin/env python
"""
Script to analyze misclassifications and their relationship to annotator uncertainty.
Takes a JSON file of misclassifications and analyzes the correlation between
model confidence and annotator confidence.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_misclassifications(json_path):
    """Load misclassifications from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        misclassifications = json.load(f)
    return misclassifications


def normalize_scores(misclassifications):
    """
    Normalize original scores to a 0-1 scale.
    - A score of 1 maps to 0 (certain benign)
    - A score of 5 maps to 1 (certain toxic)
    """
    for item in misclassifications:
        original_score = item["ground_truth"].get("original_score")
        if original_score is not None:
            # Normalize from 1-5 scale to 0-1 scale
            normalized_score = (original_score - 1) / 4.0
            item["ground_truth"]["normalized_score"] = normalized_score
        else:
            item["ground_truth"]["normalized_score"] = None

    return misclassifications


def calculate_statistics(misclassifications):
    """Calculate statistics about the misclassifications."""
    # Extract values for analysis
    data = []
    for item in misclassifications:
        normalized_score = item["ground_truth"].get("normalized_score")
        toxic_prob = item["model_prediction"]["probabilities"][
            1
        ]  # Probability of toxic class
        binary_class = item["ground_truth"]["binary_class"]
        predicted_class = item["model_prediction"]["class"]

        if normalized_score is not None:
            data.append(
                {
                    "normalized_score": normalized_score,
                    "toxic_prob": toxic_prob,
                    "binary_class": binary_class,
                    "predicted_class": predicted_class,
                    "text": item["text"][:50] + "..."
                    if len(item["text"]) > 50
                    else item["text"],
                }
            )

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)

    # Calculate correlation between normalized score and toxic probability
    if not df.empty:
        correlation, p_value = stats.pearsonr(df["normalized_score"], df["toxic_prob"])
        spearman_corr, spearman_p = stats.spearmanr(
            df["normalized_score"], df["toxic_prob"]
        )

        # Calculate model confidence (distance from 0.5 probability)
        df["model_confidence"] = abs(df["toxic_prob"] - 0.5) * 2  # Scales to 0-1

        # Calculate annotator confidence (distance from 0.5 normalized score)
        df["annotator_confidence"] = (
            abs(df["normalized_score"] - 0.5) * 2
        )  # Scales to 0-1

        # Correlation between confidences
        confidence_corr, confidence_p = stats.pearsonr(
            df["model_confidence"], df["annotator_confidence"]
        )

        stats_dict = {
            "pearson_correlation": correlation,
            "pearson_p_value": p_value,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "confidence_correlation": confidence_corr,
            "confidence_p_value": confidence_p,
            "num_samples": len(df),
            "toxic_class_count": df[df["binary_class"] == 1].shape[0],
            "benign_class_count": df[df["binary_class"] == 0].shape[0],
        }

        return stats_dict, df

    return None, None


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

    plt.close("all")


def save_stats(stats, output_dir):
    """Save statistics to a JSON file."""
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Also create a human-readable version
    stats_txt = os.path.join(output_dir, "stats.txt")
    with open(stats_txt, "w", encoding="utf-8") as f:
        f.write("Analysis of Misclassifications\n")
        f.write("=============================\n\n")
        f.write(f"Number of samples: {stats['num_samples']}\n")
        f.write(f"Benign class samples: {stats['benign_class_count']}\n")
        f.write(f"Toxic class samples: {stats['toxic_class_count']}\n\n")

        f.write("Correlation between annotator scores and model probabilities:\n")
        f.write(
            f"  Pearson correlation: {stats['pearson_correlation']:.4f} (p-value: {stats['pearson_p_value']:.4f})\n"
        )
        f.write(
            f"  Spearman correlation: {stats['spearman_correlation']:.4f} (p-value: {stats['spearman_p_value']:.4f})\n\n"
        )

        f.write("Correlation between annotator confidence and model confidence:\n")
        f.write(
            f"  Pearson correlation: {stats['confidence_correlation']:.4f} (p-value: {stats['confidence_p_value']:.4f})\n\n"
        )

        f.write("Interpretation:\n")
        if abs(stats["pearson_correlation"]) > 0.5:
            f.write(
                "- Strong correlation between annotator scores and model probabilities\n"
            )
        elif abs(stats["pearson_correlation"]) > 0.3:
            f.write(
                "- Moderate correlation between annotator scores and model probabilities\n"
            )
        else:
            f.write(
                "- Weak correlation between annotator scores and model probabilities\n"
            )

        if abs(stats["confidence_correlation"]) > 0.5:
            f.write(
                "- Strong correlation between annotator confidence and model confidence\n"
            )
        elif abs(stats["confidence_correlation"]) > 0.3:
            f.write(
                "- Moderate correlation between annotator confidence and model confidence\n"
            )
        else:
            f.write(
                "- Weak correlation between annotator confidence and model confidence\n"
            )

        if stats["pearson_p_value"] < 0.05:
            f.write(
                "- The correlation between scores and probabilities is statistically significant\n"
            )
        else:
            f.write(
                "- The correlation between scores and probabilities is NOT statistically significant\n"
            )

        if stats["confidence_p_value"] < 0.05:
            f.write(
                "- The correlation between confidence levels is statistically significant\n"
            )
        else:
            f.write(
                "- The correlation between confidence levels is NOT statistically significant\n"
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze misclassifications and their relation to annotator uncertainty"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSON file containing misclassifications",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="misclassification_analysis",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    # Load misclassifications
    misclassifications = load_misclassifications(input_path)
    print(f"Loaded {len(misclassifications)} misclassifications from {input_path}")

    # Normalize scores
    misclassifications = normalize_scores(misclassifications)

    # Calculate statistics
    stats, df = calculate_statistics(misclassifications)

    if stats is None or df is None or df.empty:
        print(
            "Error: Could not calculate statistics. Check if the input file has valid data."
        )
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    plot_results(df, stats, output_dir)

    # Save statistics
    save_stats(stats, output_dir)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
