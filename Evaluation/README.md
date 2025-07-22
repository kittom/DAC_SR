# Symbolic Regression Results Evaluation

This directory contains the evaluation system for analyzing and visualizing symbolic regression results.

## Overview

The evaluation system reads results from `results.csv` or `results_rounding.csv` files, standardizes variable names, creates visualizations, and calculates metrics for each discovered equation.

## Features

- **Automatic Variable Standardization**: Converts various variable formats (x0, x1, x_0, x_1, k, n) to standard format
- **Multi-dimensional Plotting**: Creates appropriate visualizations for 1D, 2D, and higher-dimensional data
- **Algorithm Labeling**: Uses algorithm names as plot labels instead of equations
- **Automatic Output Organization**: Creates `results/` or `rounded_results/` subdirectories based on input type

## Setup

The evaluation environment is automatically created by the main `setup.sh` script. If you need to create it manually:

```bash
# Create the evaluation environment
conda create -n evaluation python=3.10 -y

# Install requirements
conda activate evaluation
pip install -r Evaluation/requirements.txt
conda deactivate
```

## Usage

### Using the Shell Script (Recommended)

```bash
# Evaluate continuous results
Evaluation/run_evaluation.sh DataSets/Ground_Truth/LeadingOnes/continuous/results.csv

# Evaluate rounding results
Evaluation/run_evaluation.sh DataSets/Ground_Truth/LeadingOnes/discrete/results_rounding.csv
```

### Using Python Directly

```bash
# Activate the evaluation environment
conda activate evaluation

# Run evaluation
python Evaluation/evaluate_results.py <path_to_results_file>

# Deactivate environment
conda deactivate
```

## Input Format

The evaluation system expects:

1. **Results File**: A CSV file with algorithm names as columns and equations in the first row
2. **Original Data**: A CSV file in the same directory (e.g., `GTLeadingOnes.csv`) with features in all columns except the last, and target in the last column

### Example Results File Format

```csv
DeepSR,qlattice,kan,pysr
x1/(x2 + 1.0),0.000136 - 3.52/((3.0 - 0.1*n)*(-0.176*k - 0.176)),0.2701*x_1 - 5.1952*log(1.1554*x_2 + 8.3805) + 12.9263 - 0.7597/(3.2835*x_2 - 0.1398),x0 / (x1 - -1.0)
```

## Output

The evaluation system creates:

1. **Individual Plots**: One PNG file per algorithm showing the equation visualization
2. **Output Directory**: 
   - `results/` for standard results
   - `rounded_results/` for rounding-enabled results

### Plot Types

- **1D Data**: Line plots showing y vs x0
- **2D Data**: Comprehensive 4-panel visualization including:
  - 3D surface plot showing y as a function of x0 and x1
  - Line plot showing y vs x0 with x1 fixed at mean value
  - Line plot showing y vs x1 with x0 fixed at mean value
  - Contour plot for reference showing y values as color gradient
- **Higher Dimensions**: Text-based plots showing the equation (2D projections planned for future)

## Variable Standardization

The system automatically standardizes variable names:

- `x_0`, `x_1`, `x_2`, ... → `x0`, `x1`, `x2`, ...
- `k`, `n` → `x0`, `x1` (in order of appearance)
- Maintains mathematical consistency

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- sympy
- scikit-learn
- scipy
- plotly
- kaleido

## Future Enhancements

- [ ] MSE, R², and complexity score calculations
- [ ] 2D projections for higher-dimensional data
- [ ] Interactive plots using Plotly
- [ ] Comparative analysis between algorithms
- [ ] Statistical significance testing
- [ ] Export to various formats (PDF, SVG, etc.)

## Troubleshooting

### Common Issues

1. **"conda: command not found"**: Use the full path: `~/miniconda3/bin/conda`
2. **"Original data file not found"**: Ensure the original dataset is in the same directory as the results file
3. **"Error plotting equation"**: Some equations may be too complex or contain unsupported functions

### Debug Mode

For detailed debugging, you can modify the script to add more verbose output or run it directly in Python with additional print statements. 