# Experiment Organization System

This document describes the new organized approach to managing symbolic regression experiments.

## Directory Structure

```
DAC_SR/
├── Configs/                          # Configuration files for experiments
│   ├── config_template.json         # Template for creating new configs
│   ├── library_experiment_config.json
│   ├── control_experiment.json
│   └── rounding_experiment.json
├── Experiments/                      # All experiment results
│   └── SuiteA/                      # Experiment suites
│       ├── LibraryAlgorithmsLeadingOnesOneMax/
│       ├── ControlAlgorithmsLeadingOnesOneMax/
│       └── RoundingAlgorithmsLeadingOnesOneMax/
├── experiment_runner.py             # Single experiment runner
├── master_experiment_runner.py      # Master runner for multiple experiments
├── run_global.sh                    # Global execution script
└── global_TODO.txt                  # Consolidated TODO list
```

## Components

### 1. Configuration Files (`Configs/`)

Each experiment is defined by a JSON configuration file that specifies:
- **Experiment metadata**: name, description, suite
- **Data generation**: which benchmarks, instance sizes, data types
- **Evaluation**: which analysis styles (control, library, rounding)
- **Output settings**: directories, logging
- **Execution settings**: parallel jobs, timeouts

#### Example Config Structure:
```json
{
    "experiment": {
        "name": "MyExperiment",
        "description": "Description of the experiment",
        "suite": "SuiteA"
    },
    "data_generation": {
        "benchmarks": {
            "leadingones": {
                "enabled": true,
                "instance_sizes": [10, 20, 30],
                "data_type": "continuous"
            }
        },
        "noise_level": "1e-12",
        "dropout": 0.0
    },
    "evaluation": {
        "analysis_styles": {
            "control": {"enabled": true},
            "library": {"enabled": false},
            "rounding": {"enabled": false}
        }
    }
}
```

### 2. Single Experiment Runner (`experiment_runner.py`)

Handles individual experiments:
- Sets up experiment directories
- Generates data based on config
- Creates TODO.txt with algorithm commands
- Supports phase-based execution (`setup`, `generate`, `evaluate`, `analyze`, `all`)

### 3. Master Experiment Runner (`master_experiment_runner.py`)

Orchestrates multiple experiments:
- Reads all config files from `Configs/`
- Sets up all experiments
- Generates data for all experiments
- Creates individual TODO.txt files
- Consolidates all TODO lists into `global_TODO.txt`

### 4. Global Execution Script (`run_global.sh`)

Executes all experiments from the global TODO list:
- Uses GNU Parallel for efficient execution
- Supports configurable number of parallel jobs
- Provides progress tracking and job logging

## Usage

### Setting Up Multiple Experiments

1. **Create config files** in the `Configs/` directory:
   ```bash
   cp Configs/config_template.json Configs/my_experiment.json
   # Edit my_experiment.json with your settings
   ```

2. **Run the master experiment runner**:
   ```bash
   python master_experiment_runner.py
   ```

3. **Execute all experiments**:
   ```bash
   ./run_global.sh -j 16  # Use 16 parallel jobs
   ```

### Individual Experiment Management

For single experiments, you can still use the original approach:

```bash
# Set up and generate data
python experiment_runner.py Configs/my_experiment.json --phase generate

# Create TODO list
python experiment_runner.py Configs/my_experiment.json --phase evaluate

# Run individual experiment
cd Experiments/SuiteA/MyExperiment
./run.sh 16
```

## Analysis Styles

The system supports three analysis styles:

1. **Control**: Default SR algorithms (DeepSR, PySR, KAN, Q-Lattice, E2E Transformer, TPSR, Linear)
2. **Library**: SR algorithms with library configurations (DeepSR, PySR, TPSR, Linear)
3. **Rounding**: SR algorithms with rounding support (DeepSR, PySR, TPSR, Linear)

## Benchmarks

Supported benchmarks:
- **LeadingOnes**: Binary optimization problem
- **OneMax**: Binary optimization problem  
- **PSA-CMA-ES**: Continuous optimization problems (sphere, rosenbrock, etc.)
- **Model**: Custom model-based problems

## Parallel Execution

The system uses GNU Parallel for efficient execution:
- Automatically detects available CPU cores
- Configurable number of parallel jobs
- Progress tracking and job logging
- Graceful handling of failures

## Logging

- Individual experiment logs: `Experiments/SuiteA/ExperimentName/experiment.log`
- Master runner log: `master_experiment_runner.log`
- Global execution log: `global_experiment_joblog.txt`

## Benefits

1. **Organization**: Clear separation of configs, experiments, and execution
2. **Scalability**: Easy to add new experiments by creating config files
3. **Reproducibility**: All experiment settings captured in config files
4. **Efficiency**: Parallel execution across multiple experiments
5. **Flexibility**: Support for different analysis styles and benchmarks
6. **Maintainability**: Centralized management of experiment pipeline 