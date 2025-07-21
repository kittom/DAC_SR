#!/bin/bash

# Script to run e2e_Transformer analysis on GTLeadingOnes.csv
# Make sure to activate the correct conda environment first

echo "=== e2e_Transformer Analysis Runner ==="
echo "Activating conda environment: e2e_transformer"
echo ""

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate e2e_transformer

# Check if environment is activated
if [ "$CONDA_DEFAULT_ENV" != "e2e_transformer" ]; then
    echo "ERROR: Failed to activate e2e_transformer environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "Environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo ""

# Check if required files exist
if [ ! -f "model1.pt" ]; then
    echo "ERROR: model1.pt not found in current directory"
    exit 1
fi

if [ ! -f "analyze_leading_ones.py" ]; then
    echo "ERROR: analyze_leading_ones.py not found in current directory"
    exit 1
fi

echo "Required files found. Starting analysis..."
echo ""

# Run the analysis
python analyze_leading_ones.py

echo ""
echo "=== Analysis Complete ==="
echo "Check the generated files for results:"
echo "- leading_ones_results_*.json (detailed results)"
echo "- leading_ones_report_*.txt (summary report)"
echo "- leading_ones_analysis_plots.png (visualizations)" 