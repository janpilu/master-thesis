# Master Thesis: Toxic Language Detection using Transformer Models

A comprehensive machine learning pipeline for detecting toxic language in text using state-of-the-art transformer models. This project includes hyperparameter optimization, cross-validation evaluation, misclassification analysis, and interactive tools for model development and evaluation.

## Features

- **Multiple Transformer Models**: Support for BERT, RoBERTa, DeBERTa, and other transformer architectures
- **Hyperparameter Search**: Automated grid search and parameter optimization
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Misclassification Analysis**: Deep dive into model errors with statistical analysis
- **Interactive CLI**: User-friendly command-line interface for all operations
- **Web Interface**: Streamlit-based frontend for model interaction
- **Statistical Evaluation**: Comprehensive statistical tests and comparisons
- **Annotation Tools**: Built-in tools for data annotation and evaluation

## 📁 Project Structure

```
master-thesis/
├── src/                          # Core source code
│   ├── data/                     # Data loading and processing
│   ├── models/                   # Model architectures and factory
│   ├── training/                 # Training loops and utilities
│   └── utils/                    # Configuration and utility functions
├── config/                       # Configuration files (.yaml)
├── runs/                         # Training run outputs
├── misclassifications/           # Misclassification analysis results
├── plots/                        # Generated visualizations
├── scripts/                      # Utility scripts
├── cli.py                        # Interactive command-line interface
├── main.py                       # Basic training script
├── hyper_main.py                 # Hyperparameter search
├── frontend.py                   # Streamlit web interface
├── evaluate_*.py                 # Evaluation scripts
└── analyze_*.py                  # Analysis scripts
```

## 🔧 Installation

### Prerequisites

- Python 3.12 or higher
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd master-thesis

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd master-thesis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Interactive CLI (Recommended)

The easiest way to get started is using the interactive CLI:

```bash
python cli.py
```

This provides a menu-driven interface for:

- Starting new training runs
- Hyperparameter search
- Cross-validation
- Misclassification analysis
- Result visualization

### 2. Basic Training

Train a single model with a configuration file:

```bash
python main.py
```

This will train models using all configuration files in the `config/` directory.

### 3. Web Interface

Launch the Streamlit web interface:

```bash
python frontend.py
```

Navigate to `http://localhost:8501` to interact with trained models.

## 📊 Usage Examples

### Hyperparameter Search

1. **Using the CLI**:

   ```bash
   python cli.py
   # Select "Start new hyperparameter search"
   ```

2. **Command Line**:

   ```bash
   python hyper_main.py --config config/your_config.yaml \
                        --param-variations "training.learning_rate=1e-5,2e-5,5e-5" \
                        --run-prefix "lr_experiment"
   ```

3. **Config File Variations**:
   ```yaml
   # In your config.yaml
   training:
     learning_rate: [1e-5, 2e-5, 5e-5]
     batch_size: [16, 32]
   ```

### Cross-Validation

Run k-fold cross-validation for robust evaluation:

```bash
python cli.py
# Select "Start cross-validation"
```

### Misclassification Analysis

Analyze model errors and generate insights:

```bash
python cli.py
# Select "Evaluate misclassifications"
```

Or run directly:

```bash
python evaluate_misclassifications.py <model_path> <data_path>
```

## 📋 Configuration

Create YAML configuration files in the `config/` directory. Example:

```yaml
model:
  name: "microsoft/deberta-v3-base"
  classification_head:
    architecture:
      - type: linear
        out_features: 512
      - type: dropout
        p: 0.1
      - type: linear
        out_features: 1

training:
  learning_rate: 2e-5
  batch_size: 16
  num_epochs: 5
  weight_decay: 0.01

data:
  dataset_name: "toxigen"
  max_length: 512
  train_split: "train"
  test_split: "test"
```

## 🔬 Evaluation and Analysis

### Statistical Evaluation

The project includes comprehensive statistical analysis tools:

- **Wilcoxon Signed-Rank Tests**: Compare model performance across folds
- **Cross-Validation Analysis**: Robust performance estimation
- **Misclassification Patterns**: Identify systematic errors

### Visualization

Automatic generation of:

- Performance plots
- Confusion matrices
- Loss curves
- Misclassification analysis charts

## 📁 Data

The project is designed to work with the ToxiGen dataset but can be adapted for other toxic language detection datasets. Ensure your data follows the expected format:

- Text samples with binary toxicity labels
- Support for train/validation/test splits
- Compatible with Hugging Face datasets format

## 👤 Author

**Jan Langela Regincos**
