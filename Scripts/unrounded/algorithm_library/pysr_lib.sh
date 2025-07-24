#!/bin/bash
set -e

# PySR Minimal Library Activation and Execution Script
# This script activates the pysr_env conda environment and runs PySR with minimal library

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
echo "PySR Minimal Library Symbolic Regression"
echo "=========================================="
echo "CSV File: $CSV_FILE"
echo "Problem Type: $PROBLEM_TYPE"
echo "Noise Parameter: $NOISE"
echo "=========================================="

# Change to PySR directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYSR_DIR="$SCRIPT_DIR/../../SR_algorithms/PySR"

if [ ! -d "$PYSR_DIR" ]; then
    echo "Error: PySR directory not found at $PYSR_DIR"
    exit 1
fi

cd "$PYSR_DIR"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating pysr_env conda environment..."
conda activate pysr_env

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'pysr_env'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"
echo "Problem type: $PROBLEM_TYPE"
echo "Noise parameter: $NOISE"

echo "Running PySR with minimal library on CSV data..."
python run_pysr_lib.py "$CSV_FILE" "$PROBLEM_TYPE" --noise "$NOISE"

if [ $? -eq 0 ]; then
    echo "PySR minimal library execution completed successfully!"
else
    echo "Error: PySR minimal library execution failed!"
    exit 1
fi 