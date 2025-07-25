# Symbolic Regression Results Analysis

This directory contains the analysis system for analyzing and visualizing symbolic regression results.

## Overview
The analysis system reads results from `results.csv` or `results_rounding.csv` files, standardizes variable names, creates visualizations, and calculates metrics for each discovered equation.

## Environment Setup
The analysis environment is automatically created by the main `setup.sh` script. If you need to create it manually:

```bash
# Create the evaluation environment
conda create -n evaluation python=3.10 -y
conda activate evaluation
pip install -r Analysis/requirements.txt
```

## Usage

```bash
Analysis/run_analysis.sh DataSets/Ground_Truth/LeadingOnes/continuous/results.csv
```

or for rounding analysis:

```bash
Analysis/run_analysis.sh DataSets/Ground_Truth/LeadingOnes/discrete/results_rounding.csv
```

### Manual Usage

```bash
# Activate the evaluation environment
conda activate evaluation
# Run analysis
python Analysis/analyze_results.py <path_to_results_file>
```

## Input Format
The analysis system expects:
- A CSV file with columns for discovered equations and ground truth
- Optionally, additional columns for algorithm metadata

## Output
The analysis system creates:
- Plots and summary statistics for each equation
- Comparison tables for all algorithms
- Output files in the same directory as the input CSV

## Directory Structure
- `analyze_results.py`: Main analysis script
- `run_analysis.sh`: Shell script for running analysis
- `requirements.txt`: Python dependencies
- `README.md`: This file 