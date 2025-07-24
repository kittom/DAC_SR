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

# Run all symbolic regression algorithms with minimal library
bash "$PROJECT_ROOT/Scripts/run_all_library.sh" "$PROBLEM_TYPE" "$DATASET_PATH" "$NOISE_LEVEL"

echo "=========================================="
echo "Tailored library evaluation completed!"
echo "Results saved to: $(dirname "$DATASET_PATH")/results_lib.csv"
echo "==========================================" 