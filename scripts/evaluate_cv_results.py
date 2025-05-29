import glob
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests


def extract_metrics(history_file):
    """Extract metrics from history.json file."""
    with open(history_file, "r") as f:
        history = json.load(f)
        val_loss = history.get("val_loss", [])

        # Metrics might be in a nested structure
        metrics = history.get("metrics", {})
        val_accuracy = metrics.get("accuracy", [])
        val_f1_score = metrics.get("f1_score", [])

        return {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_score": val_f1_score,
        }


def get_config_name(config_dir):
    """Extract learning rate, batch size, and frozen status from config file to create a descriptive name."""
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            lr = config.get("training", {}).get("learning_rate", "unknown")
            bs = config.get("training", {}).get("batch_size", "unknown")
            frozen = config.get("model", {}).get("freeze_bert", False)
            name = f"LR={lr}; BS={bs}"
            if frozen:
                name += "; Frozen"
            return name
    return "unknown"


def get_best_metrics(cv_dir, config_num):
    """Get best metrics for each fold in a CV run for a specific config."""
    config_dir = os.path.join(cv_dir, f"config_{config_num}")
    all_losses = []
    all_accuracies = []
    all_f1_scores = []

    if not os.path.exists(config_dir):
        print(f"Directory not found: {config_dir}")
        return {"loss": [], "accuracy": [], "f1_score": []}

    fold_dirs = glob.glob(os.path.join(config_dir, "fold_*"))
    for fold_dir in fold_dirs:
        history_file = os.path.join(fold_dir, "history.json")
        if os.path.exists(history_file):
            metrics = extract_metrics(history_file)
            val_losses = metrics["val_loss"]
            val_accs = metrics["val_accuracy"]
            val_f1s = metrics["val_f1_score"]

            if val_losses and val_accs and val_f1s:
                # Find index of best validation loss
                best_idx = np.argmin(val_losses)
                all_losses.append(val_losses[best_idx])
                all_accuracies.append(val_accs[best_idx])
                all_f1_scores.append(val_f1s[best_idx])

    return {"loss": all_losses, "accuracy": all_accuracies, "f1_score": all_f1_scores}


def create_bee_swarm_plot(
    data_dict,
    metric="loss",
    title="Validation Loss",
    output_dir=None,
    single_folder=False,
):
    """Create bee swarm plot with box plots for specified metric. Optionally annotate min, max, median if single_folder is True."""
    plt.figure(figsize=(14, 6))  # Make figure wider

    # Prepare data for plotting
    labels = []
    values = []
    for model, metrics in data_dict.items():
        if metrics[metric]:
            labels.extend([model] * len(metrics[metric]))
            values.extend(metrics[metric])

    # Create DataFrame for seaborn
    value_col = "Value"
    df = pd.DataFrame({"Model": labels, value_col: values})

    # Create a subplot with enough margin on right side for annotations
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(right=0.82)  # Adjust right margin to make room for annotations

    # Calculate figure width and adjust the box spacing
    n_boxes = len(df["Model"].unique())
    # Reduce padding between boxes (smaller width means more space between them)
    box_width = 0.4
    # Use matplotlib's boxplot for more control over spacing
    sns.boxplot(
        data=df,
        x="Model",
        y=value_col,
        color="lightgray",
        width=box_width,
        ax=ax,
        saturation=1,
        fliersize=0,
    )  # No outliers (they'll be shown in swarmplot)

    # Compress the x-axis range to bring boxes closer
    if n_boxes > 1:
        x_min, x_max = ax.get_xlim()
        padding = (x_max - x_min) * 0.15  # Reduce padding by 15%
        ax.set_xlim(x_min - padding / 2, x_max - padding / 2)

    sns.swarmplot(data=df, x="Model", y=value_col, color="red", size=6, ax=ax)

    # Remove the figure we created earlier as we're now using the new fig, ax
    plt.close()

    # Use the new axes for remaining operations
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(f"Box Plot and Bee Swarm Plot of {title}")

    # Set appropriate y-axis label based on metric
    if metric == "loss":
        ax.set_ylabel("Validation Loss")
    elif metric == "accuracy":
        ax.set_ylabel("Validation Accuracy")
    elif metric == "f1_score":
        ax.set_ylabel("F1 Score")

    # Set y-axis limits for accuracy and F1 score
    if metric in ["accuracy", "f1_score", "loss"]:
        ax.set_ylim(0, 1)

    # Annotate min, max, median if only one folder is selected
    if single_folder:
        models = df["Model"].unique()
        # Get vertical center of the plot for annotation alignment
        y_min, y_max = ax.get_ylim()
        plot_y_center = (y_min + y_max) / 2

        for i, model in enumerate(models):
            vals = df[df["Model"] == model][value_col].values
            if len(vals) > 0:
                min_val = np.min(vals)
                max_val = np.max(vals)
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                # Center annotation vertically next to the box
                y_center = plot_y_center
                annotation = f"min={min_val:.3f}\nmax={max_val:.3f}\nmean={mean_val:.3f}\nstd={std_val:.3f}"
                x_offset = box_width / 3 + 0.13
                ax.text(
                    i + x_offset,
                    y_center,
                    annotation,
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="gray",
                        boxstyle="round,pad=0.2",
                    ),
                )

    fig.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"cv_results_{metric}_bee_swarm.png"))
    else:
        fig.savefig(f"cv_results_{metric}_bee_swarm.png")
    plt.close(fig)


def create_wilcoxon_matrix(
    data_dict, metric="loss", title="Validation Loss", output_dir=None
):
    """Create Wilcoxon test matrix for specified metric."""
    models = [model for model, metrics in data_dict.items() if metrics[metric]]
    n_models = len(models)
    p_values = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                try:
                    _, p_value = stats.wilcoxon(
                        data_dict[models[i]][metric], data_dict[models[j]][metric]
                    )
                    p_values[i, j] = p_value
                except:
                    p_values[i, j] = np.nan

    # Create DataFrame for visualization
    df = pd.DataFrame(p_values, index=models, columns=models)

    # Custom colormap: white for >=0.5, red for <0.5, with exponential steps and a 0.05 step
    colors = [
        (1, 0, 0),  # intense red for p < 0.001
        (1, 0.4, 0.4),  # lighter red for p < 0.005
        (1, 0.7, 0.7),  # even lighter red for p < 0.01
        (1, 0.85, 0.85),  # very light red for p < 0.05
        (1, 1, 1),  # white for p < 0.5
        (1, 1, 1),  # white for p >= 0.5
    ]
    boundaries = [0, 0.001, 0.005, 0.01, 0.05, 0.5, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_red", colors, N=256)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        norm=norm,
        mask=np.isnan(p_values),
        cbar_kws={"label": "p-value"},
    )
    plt.title(f"Wilcoxon Test P-Values Matrix ({title})")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"cv_results_{metric}_wilcoxon.png"))
    else:
        plt.savefig(f"cv_results_{metric}_wilcoxon.png")
    plt.close()

    return p_values, models


def create_bh_corrected_matrix(
    p_values, models, metric="loss", title="Validation Loss", output_dir=None
):
    """Create Benjamini-Hochberg corrected matrix for specified metric."""
    # Flatten p-values and apply BH correction
    flat_p_values = p_values[p_values > 0]  # Exclude diagonal zeros
    if len(flat_p_values) > 0:  # Only apply correction if there are p-values
        _, corrected_p_values, _, _ = multipletests(flat_p_values, method="fdr_bh")

        # Reconstruct matrix
        n = p_values.shape[0]
        corrected_matrix = np.zeros((n, n))
        mask = p_values > 0
        corrected_matrix[mask] = corrected_p_values

        # Create DataFrame for visualization
        df = pd.DataFrame(corrected_matrix, index=models, columns=models)

        # Custom colormap: white for >=0.5, red for <0.5, with exponential steps and a 0.05 step
        colors = [
            (1, 0, 0),  # intense red for p < 0.001
            (1, 0.4, 0.4),  # lighter red for p < 0.005
            (1, 0.7, 0.7),  # even lighter red for p < 0.01
            (1, 0.85, 0.85),  # very light red for p < 0.05
            (1, 1, 1),  # white for p < 0.5
            (1, 1, 1),  # white for p >= 0.5
        ]
        boundaries = [0, 0.001, 0.005, 0.01, 0.05, 0.5, 1]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_red", colors, N=256)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df,
            annot=True,
            fmt=".4f",
            cmap=cmap,
            norm=norm,
            mask=np.isnan(corrected_matrix),
            cbar_kws={"label": "p-value"},
        )
        plt.title(f"Benjamini-Hochberg Corrected P-Values Matrix ({title})")
        plt.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, f"cv_results_{metric}_bh_corrected.png")
            )
        else:
            plt.savefig(f"cv_results_{metric}_bh_corrected.png")
        plt.close()


def create_cohens_d_matrix(
    data_dict, metric="loss", title="Validation Loss", output_dir=None
):
    """Create Cohen's d effect size matrix for specified metric."""
    models = [model for model, metrics in data_dict.items() if metrics[metric]]
    n_models = len(models)
    d_values = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                try:
                    n1, n2 = (
                        len(data_dict[models[i]][metric]),
                        len(data_dict[models[j]][metric]),
                    )
                    var1, var2 = (
                        np.var(data_dict[models[i]][metric], ddof=1),
                        np.var(data_dict[models[j]][metric], ddof=1),
                    )
                    pooled_se = np.sqrt(
                        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                    )
                    d = (
                        np.mean(data_dict[models[i]][metric])
                        - np.mean(data_dict[models[j]][metric])
                    ) / pooled_se
                    d_values[i, j] = d
                except:
                    d_values[i, j] = np.nan

    # Create DataFrame for visualization
    df = pd.DataFrame(d_values, index=models, columns=models)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, mask=np.isnan(d_values)
    )
    plt.title(f"Cohen's d Effect Size Matrix ({title})")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"cv_results_{metric}_cohens_d.png"))
    else:
        plt.savefig(f"cv_results_{metric}_cohens_d.png")
    plt.close()


def create_metric_visualizations(
    data_dict,
    metric="loss",
    title="Validation Loss",
    output_dir=None,
    single_folder=False,
    show_p_value=True,
):
    """Create all visualizations for a specific metric."""
    if single_folder:
        # Perform Friedman test across configurations
        models = [model for model, metrics in data_dict.items() if metrics[metric]]
        data = [data_dict[model][metric] for model in models]
        try:
            stat, p_value = stats.friedmanchisquare(*data)
        except Exception:
            p_value = np.nan

        # Conditionally update title to include p-value if show_p_value is True
        if show_p_value:
            title_with_p = f"{title} (Friedman p = {p_value:.4f})"
            create_bee_swarm_plot(
                data_dict, metric, title_with_p, output_dir, single_folder=single_folder
            )
        else:
            create_bee_swarm_plot(
                data_dict, metric, title, output_dir, single_folder=single_folder
            )
    else:
        # Create bee swarm plot
        create_bee_swarm_plot(
            data_dict, metric, title, output_dir, single_folder=single_folder
        )

        # Create Wilcoxon test matrix and get p-values for BH correction
        p_values, models = create_wilcoxon_matrix(data_dict, metric, title, output_dir)

        # Create BH corrected matrix
        create_bh_corrected_matrix(p_values, models, metric, title, output_dir)

        # Create Cohen's d matrix
        create_cohens_d_matrix(data_dict, metric, title, output_dir)


def main():
    # Interactive selection of CV run directories
    cv_runs = {}
    from pathlib import Path

    import questionary

    # List available CV directories
    cv_base_dirs = sorted(
        [d for d in Path("runs").iterdir() if d.is_dir() and d.name.startswith("cv_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not cv_base_dirs:
        print("No cross-validation directories found in 'runs'.")
        return

    # Prompt user to select runs
    selected_runs = questionary.checkbox(
        "Select cross-validation directories to process:",
        choices=[d.name for d in cv_base_dirs],
    ).ask()
    if not selected_runs:
        print("No directories selected. Exiting.")
        return

    # Detect if only one folder is selected
    single_folder = len(selected_runs) == 1

    # Ask if p-value should be shown in title (only relevant for single folder)
    show_p_value = True  # Default
    if single_folder:
        show_p_value = questionary.confirm(
            "Would you like to show the Friedman test p-value in the plot titles?",
            default=True,
        ).ask()

    print("Processing data from selected CV runs...")

    # Process each selected CV run
    for run_name in selected_runs:
        cv_dir = os.path.join("runs", run_name)
        plots_dir = os.path.join(cv_dir, "plots")
        config_dirs = sorted(glob.glob(os.path.join(cv_dir, "config_*")))
        for config_path in config_dirs:
            config_num = int(os.path.basename(config_path).split("_")[1])
            config_name = get_config_name(config_path)
            model_name = config_name  # Only use config_name, no cv_xxxx

            print(f"Processing {model_name}...")
            metrics = get_best_metrics(cv_dir, config_num)
            if metrics["loss"]:
                cv_runs[model_name] = metrics
                print(f"  Found {len(metrics['loss'])} samples")
            else:
                print("No data found")

    if not cv_runs:
        print("No data found in the specified directories")
        return

    print("Creating visualizations...")

    print(cv_runs)

    # Create visualizations for each metric
    metrics_info = [
        ("loss", "Validation Loss"),
        ("accuracy", "Validation Accuracy"),
        ("f1_score", "F1 Score"),
    ]
    for metric, title in metrics_info:
        create_metric_visualizations(
            cv_runs,
            metric=metric,
            title=title,
            output_dir=plots_dir,
            single_folder=single_folder,
            show_p_value=show_p_value,
        )

    print("Done!")


if __name__ == "__main__":
    main()
