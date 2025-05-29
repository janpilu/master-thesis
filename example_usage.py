#!/usr/bin/env python3
"""
Example usage of the Wilcoxon cross-validation comparison script.
This creates sample data and demonstrates the script functionality.
"""

import numpy as np
import pandas as pd
from wilcoxon_crossval_comparison import create_comparison_matrix, print_results_summary


def create_sample_data(filename: str, performance_boost: float = 0.0) -> None:
    """Create sample cross-validation data with optional performance boost."""
    np.random.seed(42)  # For reproducibility

    data = []
    configs = [1, 2, 3]
    folds = list(range(1, 11))  # 10 folds

    for config in configs:
        for fold in folds:
            # Base performance with some variation
            base_val_loss = 0.45 + np.random.normal(0, 0.05)
            base_val_acc = 0.75 + np.random.normal(0, 0.02)
            base_val_f1 = 0.65 + np.random.normal(0, 0.03)

            # Apply performance boost (positive for finetuned, 0 for frozen)
            val_loss = base_val_loss - performance_boost * 0.02  # Lower is better
            val_acc = base_val_acc + performance_boost * 0.02  # Higher is better
            val_f1 = base_val_f1 + performance_boost * 0.03  # Higher is better

            # Random epoch and other training params
            epoch = np.random.randint(4, 10)
            batch_size = 16 if config in [2, 3] else 32
            lr = [2e-05, 5e-06, 1e-05][config - 1]
            train_loss = val_loss - np.random.uniform(0.01, 0.05)

            data.append(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "config": config,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "train_loss": max(0.1, train_loss),  # Ensure positive
                    "val_loss": max(0.1, val_loss),  # Ensure positive
                    "val_accuracy": min(
                        0.99, max(0.5, val_acc)
                    ),  # Clamp to reasonable range
                    "val_f1_score": min(
                        0.99, max(0.3, val_f1)
                    ),  # Clamp to reasonable range
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created sample data: {filename}")


def main():
    """Demonstrate the Wilcoxon comparison script with sample data."""
    print("Creating sample data...")

    # Create frozen encoder results (baseline)
    create_sample_data("sample_frozen_results.csv", performance_boost=0.0)

    # Create finetuned encoder results (with slight improvement)
    create_sample_data("sample_finetuned_results.csv", performance_boost=1.0)

    print("\nLoading and comparing data...")

    # Load the data
    frozen_df = pd.read_csv("sample_frozen_results.csv")
    finetuned_df = pd.read_csv("sample_finetuned_results.csv")

    # Perform comparison
    results_df = create_comparison_matrix(finetuned_df, frozen_df)

    # Display results
    print_results_summary(results_df, alpha=0.05)

    # Save results
    results_df.to_csv("example_wilcoxon_results.csv")
    print("\nExample results saved to: example_wilcoxon_results.csv")

    print("\nTo use with your own data, run:")
    print("python wilcoxon_crossval_comparison.py finetuned_file.csv frozen_file.csv")


if __name__ == "__main__":
    main()
