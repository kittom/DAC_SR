# Experiment 1: Learning Ground Truth Equations for Benchmark Problems

## Overview

This experiment evaluates how well different symbolic regression models can learn ground truth equations for three benchmark problems: LeadingOnes, OneMax, and PSA-CMA-ES benchmarks.

## Structure

```
Experiment1/
├── generate_data.sh              # Modular data generation script
├── run_evaluation.sh             # Modular evaluation script
├── run_all_evaluations.sh        # Master evaluation script
├── script_utils/                 # Reusable components
│   ├── generation/               # Data generation utilities
│   │   ├── generate_onemax.sh
│   │   ├── generate_leadingones.sh
│   │   └── generate_psacmaes.sh
│   └── evaluation/               # Evaluation utilities
│       ├── evaluate_control_library.sh
│       ├── evaluate_tailored_library.sh
│       └── evaluate_rounding.sh
├── Datasets/                     # Generated datasets
│   ├── OneMax/
│   │   ├── continuous/           # Continuous data for control/tailored evaluation
│   │   └── discrete/             # Discrete data for rounding evaluation
│   ├── LeadingOnes/
│   │   ├── continuous/           # Continuous data for control/tailored evaluation
│   │   └── discrete/             # Discrete data for rounding evaluation
│   └── PSACMAES/
│       ├── continuous/           # Continuous data for control/tailored evaluation
│       └── discrete/             # Discrete data for rounding evaluation
└── experiment1_description.txt   # Detailed experiment description
```

## Data Generation

### Full Generation
Generate all datasets:
```bash
./generate_data.sh
```

### Selective Generation
Generate specific datasets:
```bash
./generate_data.sh --onemax                    # OneMax only
./generate_data.sh --leadingones               # LeadingOnes only
./generate_data.sh --psacmaes                  # PSA-CMA-ES only
./generate_data.sh --onemax --leadingones      # OneMax and LeadingOnes
```

### Available Options
- `--all`: Generate all datasets (default)
- `--onemax`: Generate OneMax datasets
- `--leadingones`: Generate LeadingOnes datasets
- `--psacmaes`: Generate PSA-CMA-ES datasets
- `--help`: Show usage information

## Evaluation

### Full Evaluation
Run all evaluation types on all datasets:
```bash
./run_evaluation.sh
```

### Selective Evaluation
Run specific evaluation types:
```bash
./run_evaluation.sh --control                  # Control library only
./run_evaluation.sh --tailored                 # Tailored library only
./run_evaluation.sh --rounding                 # Rounding only
./run_evaluation.sh --control --tailored       # Control and tailored
```

Run on specific datasets:
```bash
./run_evaluation.sh --datasets onemax          # OneMax only
./run_evaluation.sh --datasets leadingones     # LeadingOnes only
./run_evaluation.sh --datasets psacmaes        # PSA-CMA-ES only
./run_evaluation.sh --datasets all             # All datasets
```

### Available Options
- `--all`: Run all evaluation types (default)
- `--control`: Run control library evaluation
- `--tailored`: Run tailored library evaluation
- `--rounding`: Run rounding evaluation
- `--datasets DATASETS`: Specify datasets (onemax, leadingones, psacmaes, all)
- `--help`: Show usage information

## Master Evaluation Script

The `run_all_evaluations.sh` script provides a simplified interface for common evaluation scenarios:

```bash
./run_all_evaluations.sh                       # All evaluations on all datasets
./run_all_evaluations.sh --control-only        # Control library only
./run_all_evaluations.sh --tailored-only       # Tailored library only
./run_all_evaluations.sh --rounding-only       # Rounding only
./run_all_evaluations.sh --datasets onemax     # All evaluations on OneMax
```

## Results Files

Each dataset will generate three types of results:

1. **results.csv**: Control library results (full mathematical function set) - from continuous data
2. **results_lib.csv**: Tailored library results (minimal function set) - from continuous data
3. **results_rounding.csv**: Rounding-enabled results - from discrete data

## Data Types

- **Continuous Data**: Used for control and tailored library evaluations
  - OneMax: `sqrt(x1/(x1-x2))` (continuous values)
  - LeadingOnes: `x1/(x2 + 1)` (continuous values)
  - PSA-CMA-ES: `x1 * exp(x2 * (x5 - (x3 / x4)))` (continuous values)

- **Discrete Data**: Used for rounding evaluations
  - OneMax: `round(sqrt(x1/(x1-x2)))` (rounded values)
  - LeadingOnes: `round(x1/(x2 + 1))` (rounded values)
  - PSA-CMA-ES: `round(x1 * exp(x2 * (x5 - (x3 / x4))))` (rounded values)

## Datasets

### OneMax
- **Ground Truth**: `sqrt(x1/(x1-x2))`
- **Portfolio Sizes**: 10, 20, 30, 40, 50, 100, 200, 500
- **Minimal Library**: +, -, /, sqrt

### LeadingOnes
- **Ground Truth**: `x1/(x2 + 1)`
- **Portfolio Sizes**: 10, 20, 30, 40, 50, 100, 200, 500
- **Minimal Library**: +, /

### PSA-CMA-ES
- **Ground Truth**: `x1 * exp(x2 * (x5 - (x3 / x4)))`
- **Benchmarks**: sphere, ellipsoid, rastrigin, noisy_ellipsoid, schaffer, noisy_rastrigin
- **Iterations**: 1000 per benchmark
- **Minimal Library**: *, -, /, exp

## Algorithms

The following symbolic regression algorithms are evaluated:

1. **DeepSR**: Deep Symbolic Regression
2. **PySR**: Python Symbolic Regression
3. **KAN**: Kolmogorov-Arnold Networks
4. **TPSR**: Tree-based Policy Space Representation
5. **Linear Regression**: Baseline control
6. **Q-Lattice**: Quantum-inspired lattice
7. **E2E Transformer**: End-to-end transformer

## Usage Examples

### Complete Experiment
```bash
# Generate all datasets
./generate_data.sh

# Run all evaluations
./run_evaluation.sh
```

### Testing Individual Components
```bash
# Generate only OneMax data
./generate_data.sh --onemax

# Run only control library evaluation on OneMax
./run_evaluation.sh --control --datasets onemax
```

### Debugging
```bash
# Generate specific dataset
./generate_data.sh --leadingones

# Run specific evaluation type
./run_evaluation.sh --tailored --datasets leadingones
```

## Dependencies

- **Data Generation**: Requires `generation` conda environment
- **Evaluation**: Requires respective algorithm conda environments
- **Scripts**: All scripts use absolute paths for reliability

## Notes

- All scripts include comprehensive error checking and usage information
- Results are saved in the same directory as the input datasets
- The modular structure allows for easy testing and debugging of individual components
- Scripts can be run from any directory due to absolute path usage 