# Hyperparameter Search

This functionality allows you to run multiple experiments with different hyperparameter combinations using a single base configuration file. Instead of creating multiple configuration files manually, you can specify variations for one or more parameters, and the system will generate and run all combinations.

## Features

- Create variations of a single parameter (e.g., try different learning rates)
- Create a grid search by varying multiple parameters at once
- Generate separate run directories with individual configuration files
- Each generated config has only the specific values used for that run (not the entire list)
- Descriptive run folder names based on the parameters being varied
- Parameter variations can be defined directly in the config file or via command line
- Automatically uses the first config file in the config folder if none is specified
- Executes training runs by default (can be disabled with `--no-execute`)
- Includes timestamps in run folder names for chronological sorting and uniqueness

## Usage

### Defining Parameter Variations in the Config File

The easiest way to use this functionality is to specify parameter variations directly in your YAML config file using lists:

```yaml
training:
  batch_size: [16, 32, 64] # Will try different batch sizes
  learning_rate: [1e-7, 5e-7, 1e-6, 5e-6, 1e-5] # Will try different learning rates
  num_epochs: 10 # Not a list, will be fixed for all runs
```

Any parameter specified as a list with multiple values will be treated as a hyperparameter to vary. The system will create a grid search of all possible combinations.

### Command Line Interface

Use the `hyper_main.py` script to run hyperparameter searches:

```bash
# Using variations defined in the config file (will execute runs by default)
python hyper_main.py --config config/mlp_hyperparam_config.yaml --run-prefix "mlp_experiment"

# Using the first config file in the config folder (will execute runs by default)
python hyper_main.py --run-prefix "mlp_experiment"

# Generate configurations but don't execute runs
python hyper_main.py --config config/mlp_hyperparam_config.yaml --run-prefix "mlp_experiment" --no-execute

# Overriding with command line variations
python hyper_main.py --config config/mlp_config.yaml \
                     --param-variations "training.learning_rate=1e-3,1e-4,1e-5" \
                                        "training.batch_size=16,32,64" \
                     --run-prefix "mlp_experiment"
```

#### Arguments

- `--config`: Path to the base configuration file (optional, uses first config in config folder if not provided)
- `--param-variations`: (Optional) One or more parameter variations in the format `param_path=val1,val2,val3`
- `--run-prefix`: Prefix for run directory names (default: "run")
- `--no-execute`: Flag to skip executing training runs (by default, runs are executed)

### Parameter Paths

Parameter paths use dot notation to specify nested parameters in the configuration:

- `training.learning_rate`: Learning rate in the training section
- `model.classification_head.architecture.2.p`: Dropout probability in the third layer of the model architecture

### Examples

#### Parameter Variations in Config

Create a config file with parameter variations:

```yaml
# config/mlp_hyperparam_config.yaml
model:
  name: "microsoft/deberta-v3-base"
  # ...
  classification_head:
    # ...
    architecture:
      # ...
      - type: dropout
        p: [0.1, 0.2, 0.3] # Will try different dropout rates
      # ...

training:
  batch_size: [16, 32, 64] # Will try different batch sizes
  learning_rate: [1e-7, 5e-7, 1e-6, 5e-6, 1e-5] # Will try 5 different learning rates
  # ...
```

Run the hyperparameter search:

```bash
python hyper_main.py --config config/mlp_hyperparam_config.yaml --run-prefix "grid_search"
```

This will create 45 configurations (5×3×3 grid = 45 combinations) and execute training for each one.

#### Using the First Config File

If you have only one config file or want to use the first one in the config folder:

```bash
python hyper_main.py --run-prefix "default_config_search"
```

This will automatically use the first config file (alphabetically) in the config folder and execute training.

#### Generate Configurations Without Running

If you want to generate the configurations but not execute training:

```bash
python hyper_main.py --config config/mlp_hyperparam_config.yaml --run-prefix "grid_search" --no-execute
```

This will create the run directories and config files but won't execute training.

#### Single Parameter Variation via Command Line

Try different learning rates:

```bash
python hyper_main.py --config config/mlp_config.yaml \
                     --param-variations "training.learning_rate=2e-5,5e-5,1e-4" \
                     --run-prefix "lr_search"
```

This will create three configurations, each with a different learning rate, and execute training for each one.

### Programmatic Usage

You can also use the `HyperParamSearch` class directly in your code:

#### Using Parameter Variations from Config

```python
from src.utils.hyperparam_search import HyperParamSearch

# Initialize with config that contains parameter variations
search = HyperParamSearch('config/mlp_hyperparam_config.yaml')

# Automatically detect and use parameter variations from the config
run_dirs = search.run_grid_search(run_prefix='experiment')

print(f"Created {len(run_dirs)} configurations")
```

#### Explicitly Defining Parameter Variations

```python
from src.utils.hyperparam_search import HyperParamSearch

# Initialize with base config
search = HyperParamSearch('config/mlp_config.yaml')

# Define parameter variations
param_variations = {
    'training.learning_rate': [1e-3, 1e-4, 1e-5],
    'training.batch_size': [16, 32]
}

# Create run directories with configurations
run_dirs = search.run_grid_search(param_variations, run_prefix='experiment')

print(f"Created {len(run_dirs)} configurations")
```

## Output

For each combination of parameters, the hyperparameter search will:

1. Create a run directory with a descriptive name including a timestamp (e.g., `mlp_experiment_123045_learning_rate_0.001_batch_size_32`)
2. Generate a `config.yaml` file in that directory with the specific parameter values
3. Execute training using that configuration (unless `--no-execute` is specified)

The run directories are created under the `runs/` directory.

### Chronological Sorting

Both the hyperparameter search run folders and the date-based run folders include timestamps to ensure:

1. Uniqueness - multiple runs on the same day won't conflict
2. Chronological sorting - folders will be sorted by time when viewed in a file explorer

For hyperparameter search folders, the format is: `prefix_HHMMSS_param1_value1_param2_value2`
For date-based run folders, the format is: `runs/YYYY-MM-DD_HHMMSS`
