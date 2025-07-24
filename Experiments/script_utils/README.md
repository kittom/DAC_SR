# Script Utilities

This directory contains modular, reusable scripts for data generation and evaluation tasks.

## Structure

```
script_utils/
├── generation/          # Data generation scripts
│   ├── generate_onemax.sh
│   ├── generate_leadingones.sh
│   └── generate_psacmaes.sh
├── evaluation/          # Evaluation scripts
│   ├── evaluate_control_library.sh
│   ├── evaluate_tailored_library.sh
│   └── evaluate_rounding.sh
└── README.md           # This file
```

## Generation Scripts

### `generation/generate_onemax.sh`
Generates OneMax datasets with portfolio sizes: 10, 20, 30, 40, 50, 100, 200, 500.

**Usage:**
```bash
./generate_onemax.sh <output_directory>
```

### `generation/generate_leadingones.sh`
Generates LeadingOnes datasets with portfolio sizes: 10, 20, 30, 40, 50, 100, 200, 500.

**Usage:**
```bash
./generate_leadingones.sh <output_directory>
```

### `generation/generate_psacmaes.sh`
Generates PSA-CMA-ES datasets for all benchmarks (sphere, ellipsoid, rastrigin, noisy_ellipsoid, schaffer, noisy_rastrigin) plus an aggregated dataset.

**Usage:**
```bash
./generate_psacmaes.sh <output_directory>
```

## Evaluation Scripts

### `evaluation/evaluate_control_library.sh`
Runs all symbolic regression algorithms with the full mathematical function library.

**Usage:**
```bash
./evaluate_control_library.sh <dataset_path> <noise_level>
```

**Output:** `results.csv` in the dataset directory

### `evaluation/evaluate_tailored_library.sh`
Runs all symbolic regression algorithms with minimal function libraries tailored to the problem type.

**Usage:**
```bash
./evaluate_tailored_library.sh <problem_type> <dataset_path> <noise_level>
```

**Problem Types:**
- `one_max`: Uses +, -, /, sqrt functions
- `leading_ones`: Uses +, / functions  
- `psa`: Uses *, -, /, exp functions

**Output:** `results_lib.csv` in the dataset directory

### `evaluation/evaluate_rounding.sh`
Runs all symbolic regression algorithms with rounding enabled.

**Usage:**
```bash
./evaluate_rounding.sh <dataset_path> <noise_level>
```

**Output:** `results_rounding.csv` in the dataset directory

## Integration with Experiments

These scripts are designed to be called by experiment-level scripts:

- **Experiment 1** (`Experiments/Experiment1/`) uses these scripts to generate three types of results:
  - `results.csv`: Control library evaluation
  - `results_lib.csv`: Tailored library evaluation
  - `results_rounding.csv`: Rounding evaluation

## Reusability

These scripts can be used independently for:
- Testing individual components
- Creating custom experiment workflows
- Debugging specific generation or evaluation steps
- Building new experiments with different configurations

## Dependencies

- All generation scripts require the `generation` conda environment
- All evaluation scripts require the respective algorithm conda environments
- Scripts use absolute paths to ensure they work from any location 