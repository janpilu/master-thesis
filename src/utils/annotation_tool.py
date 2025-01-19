"""Interactive annotation tool for analyzing and labeling hate speech data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datasets import Dataset


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
            
        Prints:
            - Score distribution statistics for each annotation label
            - Warning if no annotations are available
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


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("toxigen/toxigen-data", "annotated")
    tool = AnnotationTool(dataset["train"])
    tool.annotate_samples(n_samples=1)
    tool.analyze_annotations()
