#!/bin/bash
set -e

# Linear Regression Activation and Execution Script (Minimal Library)
# This script activates the linear conda environment and runs linear regression
# Note: Linear regression doesn't need minimal library since it's just linear

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments!"
    echo "Usage: $0 <path_to_csv_file> <problem_type> [noise]"
    echo "Example: $0 ../../DataSets/Ground_Truth/LeadingOnes/continuous/GTLeadingOnes.csv leading_ones 0.05"
    echo "Problem types: one_max, leading_ones, psa"
    exit 1
fi

CSV_FILE="$1"
PROBLEM_TYPE="$2"
NOISE="${3:-1e-12}"

if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

# Validate problem type
if [[ "$PROBLEM_TYPE" != "one_max" && "$PROBLEM_TYPE" != "leading_ones" && "$PROBLEM_TYPE" != "psa" ]]; then
    echo "Error: Problem type must be one of: one_max, leading_ones, psa"
    exit 1
fi

echo "=========================================="
echo "Linear Regression (Minimal Library)"
echo "=========================================="
echo "CSV File: $CSV_FILE"
echo "Problem Type: $PROBLEM_TYPE"
echo "Noise Parameter: $NOISE (not used for linear regression)"
echo "=========================================="

# Change to Linear directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINEAR_DIR="$SCRIPT_DIR/../../../SR_algorithms/Linear"

if [ ! -d "$LINEAR_DIR" ]; then
    echo "Error: Linear directory not found at $LINEAR_DIR"
    exit 1
fi

cd "$LINEAR_DIR"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating linear conda environment..."
conda activate linear

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'linear'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"
echo "Problem type: $PROBLEM_TYPE"
echo "Noise parameter: $NOISE (not used for linear regression)"

echo "Running linear regression on CSV data..."
python run_linear_lib.py "$CSV_FILE" "$PROBLEM_TYPE" --noise "$NOISE"

if [ $? -eq 0 ]; then
    echo "Linear regression execution completed successfully!"
else
    echo "Error: Linear regression execution failed!"
    exit 1
fi 