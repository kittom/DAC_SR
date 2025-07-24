#!/bin/bash
set -e

# Tailored Library Evaluation Script
# This script runs all symbolic regression algorithms with minimal function libraries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Get arguments
if [ $# -lt 3 ]; then
    echo "Error: Insufficient arguments provided!"
    echo "Usage: $0 <problem_type> <dataset_path> <noise_level>"
    echo "Problem types: one_max, leading_ones, psa"
    exit 1
fi

PROBLEM_TYPE="$1"
DATASET_PATH="$2"
NOISE_LEVEL="$3"

echo "=========================================="
echo "Tailored Library Evaluation"
echo "=========================================="
echo "Problem Type: $PROBLEM_TYPE"
echo "Dataset: $DATASET_PATH"
echo "Noise Level: $NOISE_LEVEL"
echo "=========================================="

# Get the directory containing the dataset
DATASET_DIR="$(dirname "$DATASET_PATH")"
GROUND_TRUTH_FILE="$DATASET_DIR/ground_truth.csv"
RESULTS_FILE="$DATASET_DIR/results_lib.csv"

# Copy ground truth to results file if it exists
if [ -f "$GROUND_TRUTH_FILE" ]; then
    echo "Copying ground truth to results file..."
    cp "$GROUND_TRUTH_FILE" "$RESULTS_FILE"
else
    echo "Warning: Ground truth file not found at $GROUND_TRUTH_FILE"
fi

# Run all symbolic regression algorithms with minimal library
bash "$PROJECT_ROOT/Scripts/run_all_library.sh" "$PROBLEM_TYPE" "$DATASET_PATH" "$NOISE_LEVEL"

echo "=========================================="
echo "Tailored library evaluation completed!"
echo "Results saved to: $RESULTS_FILE"
echo "==========================================" 