"""Interactive annotation tool for analyzing and labeling hate speech data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datasets import Dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_curve


class AnnotationTool:
    """Tool for manual annotation and analysis of hate speech data.
    
    Provides functionality for:
    - Interactive annotation of text samples
    - Saving and loading annotation progress
    - Statistical analysis of annotations vs original scores
    - Visualization of annotation distributions
    """

    def __init__(self, dataset: Dataset, save_path: str = "annotations.json"):
        """Initialize annotation tool with dataset.
        
        Args:
            dataset: HuggingFace dataset containing text samples
            save_path: Path to save annotation progress
        """
        self.dataset = dataset.to_pandas()
        self.save_path = Path(save_path)
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> dict:
        """Load existing annotations from file."""
        if self.save_path.exists():
            with open(self.save_path, "r", encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_annotations(self):
        """Save current annotations to file."""
        with open(self.save_path, "w", encoding='utf-8') as f:
            json.dump(self.annotations, f)

    def annotate_samples(self, n_samples: int = 10):
        """Interactively annotate random text samples.
        
        Args:
            n_samples: Number of samples to annotate
        """
        # Get indices of unannotated samples
        unannotated = self.dataset.index[
            ~self.dataset.index.astype(str).isin(self.annotations.keys())
        ]

        if len(unannotated) < n_samples:
            print(f"Only {len(unannotated)} samples remaining!")
            n_samples = len(unannotated)

        if n_samples == 0:
            print("No more samples to annotate!")
            return

        # Sample random indices
        sample_indices = np.random.choice(unannotated, size=n_samples, replace=False)

        # Annotate each sample
        for idx in sample_indices:
            print("\n" + "=" * 50)
            print(f"Text: {self.dataset.loc[idx, 'text']}")

            while True:
                label = input("Is this hate speech? (1: Yes, 0: No, q: quit): ").lower()
                if label == "q":
                    self._save_annotations()
                    return
                if label in ["0", "1"]:
                    self.annotations[str(idx)] = int(label)
                    break
                print("Invalid input! Please enter 0, 1, or q.")
            print(f"Original toxicity score: {self.dataset.loc[idx, 'toxicity_human']}")

        self._save_annotations()

    def analyze_annotations(self):
        """Analyze and visualize annotation statistics compared to original scores.
        
        Generates:
            - Statistical summary of original toxicity scores grouped by annotation labels
            - Box plot comparing original scores distribution across binary labels
            - Histogram showing score distribution for each label category
            - Optimal threshold analysis for toxicity score prediction
            
        Prints:
            - Score distribution statistics for each annotation label
            - Warning if no annotations are available
            - Optimal cutoff toxicity score for binary classification
        """
        if not self.annotations:
            print("No annotations available!")
            return

        # Create DataFrame with original scores and your annotations
        analysis_data = []
        for idx, hate_label in self.annotations.items():
            original_score = self.dataset.loc[int(idx), "toxicity_human"]
            analysis_data.append(
                {"original_score": original_score, "hate_label": hate_label}
            )

        df_analysis = pd.DataFrame(analysis_data)

        # Calculate statistics
        score_distribution = df_analysis.groupby("hate_label")[
            "original_score"
        ].describe()
        print("\nScore distribution for each label:")
        print(score_distribution)

        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="hate_label", y="original_score", data=df_analysis)
        plt.title("Distribution of Original Scores vs Binary Labels")
        plt.xlabel("Your Hate Speech Label (0: No, 1: Yes)")
        plt.ylabel("Original Toxicity Score")
        plt.show()

        # Create histogram
        plt.figure(figsize=(10, 6))
        for label in [0, 1]:
            scores = df_analysis[df_analysis["hate_label"] == label]["original_score"]
            plt.hist(scores, alpha=0.5, label=f"Label {label}", bins=10)
        plt.title("Histogram of Original Scores by Binary Label")
        plt.xlabel("Original Toxicity Score")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        
        # Calculate optimal threshold
        if all(label in df_analysis["hate_label"].values for label in [0, 1]):
            self._calculate_optimal_threshold(df_analysis)
        else:
            print("Need both positive and negative examples to calculate optimal threshold")
    
    def _calculate_optimal_threshold(self, df):
        """Calculate the optimal threshold for toxicity prediction.
        
        Args:
            df: DataFrame containing 'original_score' and 'hate_label' columns
            
        Generates:
            - ROC curve visualization
            - Accuracy vs threshold curve
            - Confusion matrix at optimal threshold
        """
        # Extract scores and labels
        scores = df["original_score"].values
        labels = df["hate_label"].values
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        # Calculate accuracy at different thresholds
        accuracies = []
        for threshold in thresholds:
            predicted_labels = (scores >= threshold).astype(int)
            accuracies.append(accuracy_score(labels, predicted_labels))
        
        # Find optimal threshold
        optimal_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[optimal_idx]
        optimal_accuracy = accuracies[optimal_idx]
        
        # Plot accuracy vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, lw=2)
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                    label=f'Optimal threshold = {optimal_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.legend()
        plt.show()
        
        # Calculate confusion matrix at optimal threshold
        predicted_labels = (scores >= optimal_threshold).astype(int)
        conf_matrix = confusion_matrix(labels, predicted_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Toxic', 'Toxic'],
                    yticklabels=['Non-Toxic', 'Toxic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix at Threshold = {optimal_threshold:.2f}')
        plt.show()
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
        
        # Print results
        print(f"\nOptimal toxicity score threshold: {optimal_threshold:.4f}")
        print(f"Accuracy at optimal threshold: {optimal_accuracy:.4f}")
        print("\nAt this threshold:")
        print(f"- Scores >= {optimal_threshold:.4f} should be classified as toxic")
        print(f"- Scores < {optimal_threshold:.4f} should be classified as non-toxic")


if __name__ == "__main__":
    from datasets import load_dataset
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive annotation tool for hate speech data")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing annotations without annotating new samples")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to annotate (default: 10)")
    parser.add_argument("--save-path", type=str, default="annotations.json", help="Path to save annotations (default: annotations.json)")
    args = parser.parse_args()
    
    # Load dataset and initialize tool
    dataset = load_dataset("toxigen/toxigen-data", "annotated")
    tool = AnnotationTool(dataset["train"], save_path=args.save_path)
    
    # Run the tool based on arguments
    if not args.analyze_only:
        tool.annotate_samples(n_samples=args.samples)
    
    # Always analyze annotations
    tool.analyze_annotations()
