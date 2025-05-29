#!/usr/bin/env python3
"""
Script to compare cross-validation results between fine-tuned and frozen encoder models
using pairwise Wilcoxon signed-rank tests.
"""

import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_crossval_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load cross-validation results from CSV file."""
    return pd.read_csv(filepath)


def extract_config_metrics(df: pd.DataFrame, config: int, metric: str) -> np.ndarray:
    """Extract metric values for a specific configuration across all folds."""
    config_data = df[df["config"] == config]
    if config_data.empty:
        raise ValueError(f"No data found for config {config}")

    if metric not in config_data.columns:
        raise ValueError(f"Metric '{metric}' not found in data")

    return config_data[metric].values


def perform_wilcoxon_test(
    finetuned_values: np.ndarray, frozen_values: np.ndarray
) -> float:
    """Perform pairwise Wilcoxon signed-rank test between two sets of values."""
    if len(finetuned_values) != len(frozen_values):
        raise ValueError("Arrays must have the same length for pairwise comparison")

    # Perform Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(finetuned_values, frozen_values)
    return p_value


def get_available_configs(df: pd.DataFrame) -> List[int]:
    """Get list of available configuration IDs from the dataframe."""
    return sorted(df["config"].unique())


def get_comparison_metrics() -> List[str]:
    """Get list of metrics to compare."""
    return ["val_loss", "val_accuracy", "val_f1_score"]


def create_comparison_matrix(
    finetuned_df: pd.DataFrame, frozen_df: pd.DataFrame
) -> pd.DataFrame:
    """Create matrix of p-values from Wilcoxon tests for all configs and metrics."""
    configs = get_available_configs(finetuned_df)
    metrics = get_comparison_metrics()

    # Validate that both dataframes have the same configs
    frozen_configs = get_available_configs(frozen_df)
    if set(configs) != set(frozen_configs):
        raise ValueError("Fine-tuned and frozen data must have the same configurations")

    # Initialize results matrix
    results = {}

    for metric in metrics:
        metric_results = {}
        for config in configs:
            try:
                finetuned_values = extract_config_metrics(finetuned_df, config, metric)
                frozen_values = extract_config_metrics(frozen_df, config, metric)

                p_value = perform_wilcoxon_test(finetuned_values, frozen_values)
                metric_results[f"config_{config}"] = p_value

            except ValueError as e:
                print(
                    f"Warning: Could not process config {config}, metric {metric}: {e}"
                )
                metric_results[f"config_{config}"] = np.nan

        results[metric] = metric_results

    # Convert to DataFrame for nice formatting
    return pd.DataFrame(results)


def save_results(results_df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Save results matrix to CSV file."""
    results_df.to_csv(output_path, index=True)
    print(f"Results saved to: {output_path}")


def print_results_summary(results_df: pd.DataFrame, alpha: float = 0.05) -> None:
    """Print a summary of the results with significance indicators."""
    print("\nWilcoxon Test Results (p-values):")
    print("=" * 50)
    print(results_df.round(6))

    print(f"\nSignificant differences (p < {alpha}):")
    print("-" * 40)

    for metric in results_df.columns:
        for config in results_df.index:
            p_value = results_df.loc[config, metric]
            if not np.isnan(p_value) and p_value < alpha:
                print(f"{config} - {metric}: p = {p_value:.6f} *")


def validate_input_files(
    finetuned_path: Union[str, Path], frozen_path: Union[str, Path]
) -> None:
    """Validate that input files exist and are readable."""
    for path, name in [(finetuned_path, "fine-tuned"), (frozen_path, "frozen")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

        try:
            pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Could not read {name} file {path}: {e}")


def main():
    """Main function to orchestrate the comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare cross-validation results using Wilcoxon tests"
    )
    parser.add_argument(
        "finetuned_file", help="Path to CSV file with fine-tuned encoder results"
    )
    parser.add_argument(
        "frozen_file", help="Path to CSV file with frozen encoder results"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="wilcoxon_results.csv",
        help="Output file for results (default: wilcoxon_results.csv)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for summary (default: 0.05)",
    )

    args = parser.parse_args()

    try:
        # Validate input files
        validate_input_files(args.finetuned_file, args.frozen_file)

        # Load data
        print("Loading data...")
        finetuned_df = load_crossval_data(args.finetuned_file)
        frozen_df = load_crossval_data(args.frozen_file)

        print(f"Fine-tuned data: {len(finetuned_df)} rows")
        print(f"Frozen data: {len(frozen_df)} rows")

        # Perform comparison
        print("Performing Wilcoxon tests...")
        results_df = create_comparison_matrix(finetuned_df, frozen_df)

        # Save and display results
        save_results(results_df, args.output)
        print_results_summary(results_df, args.alpha)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
