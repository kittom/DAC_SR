#!/bin/bash
set -e

# KAN Minimal Library Activation and Execution Script
# This script activates the kan conda environment and runs KAN with minimal library

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
echo "KAN Minimal Library Symbolic Regression"
echo "=========================================="
echo "CSV File: $CSV_FILE"
echo "Problem Type: $PROBLEM_TYPE"
echo "Noise Parameter: $NOISE"
echo "=========================================="

# Change to KAN directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAN_DIR="$SCRIPT_DIR/../../SR_algorithms/pykan"

if [ ! -d "$KAN_DIR" ]; then
    echo "Error: KAN directory not found at $KAN_DIR"
    exit 1
fi

cd "$KAN_DIR"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating kan conda environment..."
conda activate kan

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'kan'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"
echo "Problem type: $PROBLEM_TYPE"
echo "Noise parameter: $NOISE"

echo "Running KAN with minimal library on CSV data..."
python main_lib.py "$CSV_FILE" "$PROBLEM_TYPE" --noise "$NOISE"

if [ $? -eq 0 ]; then
    echo "KAN minimal library execution completed successfully!"
else
    echo "Error: KAN minimal library execution failed!"
    exit 1
fi 