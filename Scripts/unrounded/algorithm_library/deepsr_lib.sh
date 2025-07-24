#!/bin/bash
set -e

# DeepSR Minimal Library Activation and Execution Script
# This script activates the dso_env conda environment and runs DeepSR with minimal library

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
echo "DeepSR Minimal Library Symbolic Regression"
echo "=========================================="
echo "CSV File: $CSV_FILE"
echo "Problem Type: $PROBLEM_TYPE"
echo "Noise Parameter: $NOISE"
echo "=========================================="

# Change to DeepSR directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSR_DIR="$SCRIPT_DIR/../../../SR_algorithms/DeepSR/deep-symbolic-optimization"

if [ ! -d "$DEEPSR_DIR" ]; then
    echo "Error: DeepSR directory not found at $DEEPSR_DIR"
    exit 1
fi

cd "$DEEPSR_DIR"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating dso_env conda environment..."
conda activate dso_env

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'dso_env'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"
echo "Problem type: $PROBLEM_TYPE"
echo "Noise parameter: $NOISE"

echo "Running DeepSR with minimal library on CSV data..."
python run_deepsr_lib.py "$CSV_FILE" "$PROBLEM_TYPE" --noise "$NOISE"

if [ $? -eq 0 ]; then
    echo "DeepSR minimal library execution completed successfully!"
else
    echo "Error: DeepSR minimal library execution failed!"
    exit 1
fi 