#!/usr/bin/env python3

import glob
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import questionary
import seaborn as sns
from questionary import Style
from rich.console import Console
from rich.table import Table
from scipy import stats

metrics_label_map = {
    "val_loss": "Validation Loss",
    "val_accuracy": "Validation Accuracy",
    "val_f1_score": "Validation F1 Score",
}

# Set up a nice style for the questionary prompts
custom_style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green bold"),
        ("separator", "fg:cyan"),
        ("instruction", "fg:white"),
        ("text", "fg:white"),
    ]
)


def get_cv_folders(base_path):
    """Get all CV folders in the base path"""
    # Get all directories with cv_ prefix
    cv_folders = [
        f for f in glob.glob(os.path.join(base_path, "cv_*")) if os.path.isdir(f)
    ]
    return cv_folders


def analyze_best_epochs(scores_file):
    """Analyze the scores.csv file to find the best epoch per fold per config"""
    # Read the scores.csv file
    df = pd.read_csv(scores_file)

    # Find the best epoch (lowest val_loss) for each fold of each config
    best_epochs = df.loc[df.groupby(["fold", "config"])["val_loss"].idxmin()]

    # Sort by config and fold for readability
    best_epochs = best_epochs.sort_values(["config", "fold"])

    return best_epochs


def cohens_d(d1, d2):
    """Calculate Cohen's d effect size between two samples"""
    # Calculate the size of samples
    n1, n2 = len(d1), len(d2)

    # Calculate the variance of the samples
    var1, var2 = np.var(d1, ddof=1), np.var(d2, ddof=1)

    # Calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    return (np.mean(d1) - np.mean(d2)) / s


def perform_statistical_tests(df1, df2, metric):
    """Perform statistical tests (Wilcoxon, t-test, and Cohen's d) for each config"""
    # Get unique configs
    configs = sorted(df1["config"].unique())

    results = []

    for config in configs:
        # Get data for this config, indexed by fold to ensure proper pairing
        data1_by_fold = df1[df1["config"] == config].set_index("fold")[metric]
        data2_by_fold = df2[df2["config"] == config].set_index("fold")[metric]

        # Find common folds
        common_folds = sorted(
            set(data1_by_fold.index).intersection(set(data2_by_fold.index))
        )

        if len(common_folds) == 0:
            print(f"⚠️ Warning: No common folds for config {config}, skipping.")
            continue

        if len(common_folds) < len(data1_by_fold) or len(common_folds) < len(
            data2_by_fold
        ):
            print(
                f"⚠️ Warning: Config {config} has different folds between the two folders."
            )
            print(f"   Using only {len(common_folds)} common folds for comparison.")

        # Extract paired data using common folds to ensure proper matching
        data1 = [data1_by_fold.loc[fold] for fold in common_folds]
        data2 = [data2_by_fold.loc[fold] for fold in common_folds]

        # Convert to numpy arrays
        data1 = np.array(data1)
        data2 = np.array(data2)

        # Calculate Wilcoxon test (paired test)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(data1, data2)

        # Also calculate paired t-test for more sensitivity to differences
        t_stat, t_p = stats.ttest_rel(data1, data2)

        # Calculate Cohen's d effect size
        d = cohens_d(data1, data2)

        # Compute means
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        # Store results
        results.append(
            {
                "config": config,
                f"{metric}_mean1": mean1,
                f"{metric}_mean2": mean2,
                f"{metric}_diff": mean1 - mean2,
                "wilcoxon_p_value": wilcoxon_p,
                "t_test_p_value": t_p,
                "cohens_d": d,
                "wilcoxon_significant": wilcoxon_p < 0.05,
                "t_test_significant": t_p < 0.05,
                "effect_size_interpretation": interpret_cohens_d(d),
                "n_folds": len(common_folds),
            }
        )

    return pd.DataFrame(results)


def interpret_cohens_d(d):
    """Interpret Cohen's d value"""
    d = abs(d)  # Take absolute value for interpretation
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def display_comparison_table(df, metric, folder1_name, folder2_name):
    """Display a comparison table for the statistical tests"""
    console = Console()

    title = f"Statistical Comparison for {metric}"
    table = Table(title=title)

    table.add_column("Config", style="cyan")
    table.add_column("Folds", style="cyan")
    table.add_column(f"{folder1_name} Mean", style="green")
    table.add_column(f"{folder2_name} Mean", style="green")
    table.add_column("Difference", style="cyan")
    table.add_column("Wilcoxon p", style="yellow")
    table.add_column("t-test p", style="yellow")
    table.add_column("Significant", style="red")
    table.add_column("Cohen's d", style="magenta")
    table.add_column("Effect Size", style="blue")

    for _, row in df.iterrows():
        # Format values
        mean1 = f"{row[f'{metric}_mean1']:.4f}"
        mean2 = f"{row[f'{metric}_mean2']:.4f}"
        diff = f"{row[f'{metric}_diff']:.4f}"
        wilcoxon_p = f"{row['wilcoxon_p_value']:.4f}"
        t_test_p = f"{row['t_test_p_value']:.4f}"
        cohens_d = f"{row['cohens_d']:.4f}"

        # Determine better model
        better = folder1_name if row[f"{metric}_diff"] > 0 else folder2_name
        if metric == "val_loss":  # For loss, lower is better
            better = folder2_name if row[f"{metric}_diff"] > 0 else folder1_name

        # Consider a result significant if either test shows significance
        is_significant = row["wilcoxon_significant"] or row["t_test_significant"]
        significant = "✓" if is_significant else "✗"

        table.add_row(
            str(row["config"]),
            str(row["n_folds"]),
            mean1,
            mean2,
            f"{diff} ({better} better)",
            wilcoxon_p,
            t_test_p,
            significant,
            cohens_d,
            row["effect_size_interpretation"],
        )

    console.print(table)


def save_comparison_results(df, output_path):
    """Save the comparison results to a CSV file"""
    df.to_csv(output_path, index=False)
    print(f"✅ Saved comparison results to {output_path}")


def create_comparison_matrix(
    results_by_metric, output_dir, folder1_name, folder2_name, first_best_epochs
):
    """
    Create color-coded matrix plots for different statistical measures.

    Args:
        results_by_metric: Dictionary with metrics as keys and results DataFrames as values
        output_dir: Directory to save the plots
        folder1_name: Name of the first folder
        folder2_name: Name of the second folder
        first_best_epochs: DataFrame with the best epochs from the first folder (used to get LR and BS)
    """
    # Create a results directory for plots if it doesn't exist
    plots_dir = os.path.join(output_dir, "matrix_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Fixed title for plots
    fixed_title = (
        "10-fold Cross Validation Comparison: Fine-tuned versus frozen encoder"
    )

    # Get all unique configs across all metrics
    all_configs = set()
    for metric_results in results_by_metric.values():
        all_configs.update(metric_results["config"].astype(str))
    all_configs = sorted(all_configs)

    # Create config labels with learning rate and batch size
    config_labels = []
    for config in all_configs:
        # Extract a row for this config to get learning rate and batch size
        config_data = first_best_epochs[first_best_epochs["config"] == int(config)]
        if len(config_data) > 0:
            lr = config_data["learning_rate"].iloc[0]
            bs = config_data["batch_size"].iloc[0]
            config_labels.append(f"LR={lr}, BS={bs}")
        else:
            config_labels.append(f"{config}")

    # Get all metrics
    metrics = list(results_by_metric.keys())
    metrics_labels = [metrics_label_map[metric] for metric in metrics]

    # Create matrices for different statistical measures
    measures = {
        "p_value": {
            "title": "Wilcoxon P-Value (Lower is Better)",
            "cmap": "Reds_r",
            "vmin": 0,
            "vmax": 0.1,
        },
        "cohens_d": {
            "title": "Cohen's d (Higher is Better)",
            "cmap": "Reds",
            "vmin": 0,
            "vmax": 20.0,
        },
        "difference": {
            "title": "Absolute Difference",
            "cmap": "Reds",
            "vmin": 0,
            "vmax": None,
        },
    }

    # Create a figure for each measure
    for measure, props in measures.items():
        plt.figure(figsize=(12, 8))

        # Create the matrix
        matrix = np.zeros((len(metrics), len(all_configs)))

        # Fill the matrix
        for i, metric in enumerate(metrics):
            if metric not in results_by_metric:
                continue

            results = results_by_metric[metric]

            for j, config in enumerate(all_configs):
                row = results[results["config"].astype(str) == config]
                if len(row) == 0:
                    # Config not found for this metric
                    matrix[i, j] = np.nan
                    continue

                if measure == "p_value":
                    # Use only the Wilcoxon p-value as requested
                    value = row["wilcoxon_p_value"].values[0]
                elif measure == "cohens_d":
                    # Use absolute value for Cohen's d
                    value = abs(row["cohens_d"].values[0])
                elif measure == "difference":
                    # Use absolute difference for the difference measure
                    value = abs(row[f"{metric}_diff"].values[0])

                    # Adjust vmax for this row if not set
                    if props["vmax"] is None:
                        if i == 0:  # First metric
                            props["vmax"] = max(
                                abs(results[f"{metric}_diff"].max()), 0.001
                            )
                        else:
                            props["vmax"] = max(
                                props["vmax"], abs(results[f"{metric}_diff"].max())
                            )

                matrix[i, j] = value

        # Create heatmap
        ax = plt.subplot(111)

        # Create mask for NaN values
        mask = np.isnan(matrix)

        # Create color normalization
        norm = mcolors.Normalize(vmin=props["vmin"], vmax=props["vmax"])

        # Custom format function for p-values to ensure they're displayed correctly
        def custom_format(val):
            if measure == "p_value":
                # Truncate to 4 decimal places without rounding to match CSV values exactly
                return f"{int(val * 10000) / 10000:.4f}"
            else:
                return f"{val:.4f}"

        # Create the heatmap with custom annotation
        sns.heatmap(
            matrix,
            annot=True,
            fmt="",
            cmap=props["cmap"],
            mask=mask,
            norm=norm,
            cbar_kws={"label": props["title"]},
            xticklabels=config_labels,
            yticklabels=metrics_labels,
        )

        # Manually format the annotations
        for i in range(len(ax.texts)):
            if not mask.flatten()[i]:  # Check if the cell is not masked
                val = matrix.flatten()[i]
                ax.texts[i].set_text(custom_format(val))

        # Add a title
        plt.title(f"{fixed_title}\n{props['title']}")
        plt.ylabel("Metric")
        plt.xlabel("Configuration")

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        filename = f"matrix_{measure}_{folder1_name}_vs_{folder2_name}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()

    print(f"✅ Created matrix plots in {plots_dir}")

    # Create a combined matrix plot showing which model is better
    plt.figure(figsize=(14, 8))

    # Create the matrix for better model
    better_matrix = np.zeros((len(metrics), len(all_configs)), dtype=object)
    significant_matrix = np.zeros((len(metrics), len(all_configs)), dtype=bool)

    # Fill the matrix
    for i, metric in enumerate(metrics):
        if metric not in results_by_metric:
            continue

        results = results_by_metric[metric]

        for j, config in enumerate(all_configs):
            row = results[results["config"].astype(str) == config]
            if len(row) == 0:
                # Config not found for this metric
                better_matrix[i, j] = ""
                continue

            # Determine which model is better
            if metric == "val_loss":  # For loss, lower is better
                better = (
                    folder2_name
                    if row[f"{metric}_diff"].values[0] > 0
                    else folder1_name
                )
            else:
                better = (
                    folder1_name
                    if row[f"{metric}_diff"].values[0] > 0
                    else folder2_name
                )

            # Check if difference is significant
            is_significant = (
                row["wilcoxon_significant"].values[0]
                or row["t_test_significant"].values[0]
            )
            significant_matrix[i, j] = is_significant

            # Store the better model
            better_matrix[i, j] = better

    # Create a custom colormap for the better model matrix
    cmap = plt.cm.Blues
    custom_cmap = mcolors.ListedColormap(["lightgray", "lightblue", "orange"])

    # Create a custom annotation function to show the better model
    def annotate_better(val):
        if val == folder1_name:
            return "Fine-tuned"
        elif val == folder2_name:
            return "Frozen"
        else:
            return ""

    # Create a numerical matrix for coloring
    color_matrix = np.zeros((len(metrics), len(all_configs)))
    for i in range(len(metrics)):
        for j in range(len(all_configs)):
            if better_matrix[i, j] == "":
                color_matrix[i, j] = 0  # Gray for missing
            elif better_matrix[i, j] == folder1_name:
                color_matrix[i, j] = 1  # Blue for folder1 (fine-tuned)
            else:
                color_matrix[i, j] = 2  # Orange for folder2 (frozen)

    # Create heatmap
    ax = plt.subplot(111)
    sns.heatmap(
        color_matrix,
        annot=[[annotate_better(val) for val in row] for row in better_matrix],
        fmt="",
        cmap=custom_cmap,
        cbar=False,
        xticklabels=config_labels,
        yticklabels=metrics,
    )

    # Add bold text for significant differences
    for i in range(len(metrics)):
        for j in range(len(all_configs)):
            if significant_matrix[i, j]:
                # Get the current text object
                text = ax.texts[i * len(all_configs) + j]
                # Make it bold
                text.set_weight("bold")

    # Add a title
    plt.title(f"{fixed_title}\n(Bold = Significant Difference)")
    plt.ylabel("Metric")
    plt.xlabel("Configuration")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightblue", label="Fine-tuned"),
        Patch(facecolor="orange", label="Frozen"),
        Patch(facecolor="lightgray", label="Missing"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    filename = f"matrix_better_model_{folder1_name}_vs_{folder2_name}.png"
    plt.savefig(os.path.join(plots_dir, filename), dpi=300)
    plt.close()

    print(f"✅ Created better model matrix plot in {plots_dir}")

    return plots_dir


def main():
    # Base path for runs
    base_path = "/Users/jlangela/dev/master-thesis/runs"

    # Get all CV folders
    cv_folders = get_cv_folders(base_path)

    if len(cv_folders) < 2:
        print("❌ Need at least 2 CV folders for comparison.")
        return

    # Get folder names only (not full paths) for display
    folder_names = [os.path.basename(folder) for folder in cv_folders]

    # Ask user to select the first folder
    first_folder_name = questionary.select(
        "Select first CV folder:", choices=folder_names, style=custom_style
    ).ask()

    # Remove selected folder from list of choices for second selection
    folder_names.remove(first_folder_name)

    # Ask user to select the second folder
    second_folder_name = questionary.select(
        "Select second CV folder:", choices=folder_names, style=custom_style
    ).ask()

    # Find the full paths of the selected folders
    first_folder = next(
        (
            folder
            for folder in cv_folders
            if os.path.basename(folder) == first_folder_name
        ),
        None,
    )
    second_folder = next(
        (
            folder
            for folder in cv_folders
            if os.path.basename(folder) == second_folder_name
        ),
        None,
    )

    # Check if folders exist
    if not first_folder or not second_folder:
        print("❌ One or both selected folders not found.")
        return

    # Paths to scores.csv in the selected folders
    first_scores_file = os.path.join(first_folder, "scores.csv")
    second_scores_file = os.path.join(second_folder, "scores.csv")

    # Check if scores.csv exist
    if not os.path.exists(first_scores_file) or not os.path.exists(second_scores_file):
        print("❌ scores.csv file missing in one or both folders.")
        return

    # Analyze both folders
    first_best_epochs = analyze_best_epochs(first_scores_file)
    second_best_epochs = analyze_best_epochs(second_scores_file)

    # Check if configs match
    first_configs = set(first_best_epochs["config"].unique())
    second_configs = set(second_best_epochs["config"].unique())

    if first_configs != second_configs:
        print("⚠️ Warning: The two folders have different configurations.")
        print(f"First folder configs: {sorted(first_configs)}")
        print(f"Second folder configs: {sorted(second_configs)}")
        if not questionary.confirm("Continue anyway?", style=custom_style).ask():
            return

    # Create a comparison folder if it doesn't exist
    comparison_dir = os.path.join(base_path, "comparison_results")
    os.makedirs(comparison_dir, exist_ok=True)

    # Automatically perform tests on all metrics
    metrics = ["val_loss", "val_accuracy", "val_f1_score"]

    console = Console()

    for metric in metrics:
        console.print(f"\n[bold cyan]Analyzing {metric}[/bold cyan]")

        # Perform statistical tests
        results = perform_statistical_tests(
            first_best_epochs, second_best_epochs, metric
        )

        # Display comparison table
        display_comparison_table(results, metric, first_folder_name, second_folder_name)

        # Ask if user wants to save the results
        save_result = questionary.confirm(
            f"Do you want to save the comparison results for {metric} to a CSV file?",
            style=custom_style,
        ).ask()

        if save_result:
            # Create a filename based on the two folder names and metric
            output_filename = (
                f"comparison_{first_folder_name}_vs_{second_folder_name}_{metric}.csv"
            )
            output_path = os.path.join(comparison_dir, output_filename)

            save_comparison_results(results, output_path)

    # After analyzing all metrics, offer to save a combined report
    save_combined = questionary.confirm(
        "Do you want to save a combined summary report?", style=custom_style
    ).ask()

    if save_combined:
        # Create a summary report with key statistics from all metrics
        summary_data = []

        for metric in metrics:
            results = perform_statistical_tests(
                first_best_epochs, second_best_epochs, metric
            )

            for _, row in results.iterrows():
                config = row["config"]
                better = (
                    first_folder_name
                    if row[f"{metric}_diff"] > 0
                    else second_folder_name
                )
                if metric == "val_loss":  # For loss, lower is better
                    better = (
                        second_folder_name
                        if row[f"{metric}_diff"] > 0
                        else first_folder_name
                    )

                summary_data.append(
                    {
                        "config": config,
                        "metric": metric,
                        f"{first_folder_name}": row[f"{metric}_mean1"],
                        f"{second_folder_name}": row[f"{metric}_mean2"],
                        "difference": row[f"{metric}_diff"],
                        "better_model": better,
                        "wilcoxon_p": row["wilcoxon_p_value"],
                        "t_test_p": row["t_test_p_value"],
                        "significant": row["wilcoxon_significant"]
                        or row["t_test_significant"],
                        "cohens_d": row["cohens_d"],
                        "effect_size": row["effect_size_interpretation"],
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        summary_filename = (
            f"comparison_summary_{first_folder_name}_vs_{second_folder_name}.csv"
        )
        summary_path = os.path.join(comparison_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)

        print(f"✅ Saved combined summary to {summary_path}")

        # Also generate a more readable summary report with key findings
        report_path = os.path.join(
            comparison_dir,
            f"comparison_report_{first_folder_name}_vs_{second_folder_name}.txt",
        )

        with open(report_path, "w") as f:
            f.write(
                f"Statistical Comparison: {first_folder_name} vs {second_folder_name}\n"
            )
            f.write("=" * 80 + "\n\n")

            significant_findings = summary_df[summary_df["significant"] == True]
            f.write(f"SIGNIFICANT FINDINGS ({len(significant_findings)} total):\n")
            if len(significant_findings) > 0:
                for _, row in significant_findings.iterrows():
                    # Format p-value properly
                    p_value = min(row["wilcoxon_p"], row["t_test_p"])
                    if p_value < 0.001:
                        p_value_str = f"{p_value:.2e}"
                    else:
                        p_value_str = f"{p_value:.4f}"

                    f.write(
                        f"- Config {row['config']}, {row['metric']}: {row['better_model']} is better "
                    )
                    f.write(f"(diff: {row['difference']:.4f}, p-value: {p_value_str}, ")
                    f.write(f"effect size: {row['effect_size']})\n")
            else:
                f.write("  No statistically significant differences found.\n")

            f.write("\n\nSUMMARY BY METRIC:\n")
            for metric in metrics:
                metric_data = summary_df[summary_df["metric"] == metric]
                f.write(f"\n{metric}:\n")
                for _, row in metric_data.iterrows():
                    # Format p-value
                    p_value = min(row["wilcoxon_p"], row["t_test_p"])
                    if p_value < 0.001:
                        p_value_str = f"{p_value:.2e}"
                    else:
                        p_value_str = f"{p_value:.4f}"

                    f.write(
                        f"- Config {row['config']}: {row['better_model']} is better "
                    )
                    f.write(f"(diff: {row['difference']:.4f}, p-value: {p_value_str}, ")
                    f.write(f"significant: {'Yes' if row['significant'] else 'No'})\n")

        print(f"✅ Saved readable report to {report_path}")

    # Create a dictionary to store results by metric
    results_by_metric = {
        metric: perform_statistical_tests(first_best_epochs, second_best_epochs, metric)
        for metric in metrics
    }

    # Create comparison matrix
    create_comparison_matrix(
        results_by_metric,
        comparison_dir,
        first_folder_name,
        second_folder_name,
        first_best_epochs,
    )


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import questionary
        import rich
        import seaborn as sns
        from scipy import stats
    except ImportError:
        print("Please install the required packages:")
        print("pip install questionary pandas numpy scipy rich matplotlib seaborn")
        exit(1)

    main()
