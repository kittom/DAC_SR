#!/bin/bash
set -e

# Control Library Evaluation Script
# This script runs all symbolic regression algorithms with the full mathematical function library

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Get arguments
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments provided!"
    echo "Usage: $0 <dataset_path> <noise_level>"
    exit 1
fi

DATASET_PATH="$1"
NOISE_LEVEL="$2"

echo "=========================================="
echo "Control Library Evaluation"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Noise Level: $NOISE_LEVEL"
echo "=========================================="

# Get the directory containing the dataset
DATASET_DIR="$(dirname "$DATASET_PATH")"
GROUND_TRUTH_FILE="$DATASET_DIR/ground_truth.csv"
RESULTS_FILE="$DATASET_DIR/results.csv"

# Copy ground truth to results file if it exists
if [ -f "$GROUND_TRUTH_FILE" ]; then
    echo "Copying ground truth to results file..."
    cp "$GROUND_TRUTH_FILE" "$RESULTS_FILE"
else
    echo "Warning: Ground truth file not found at $GROUND_TRUTH_FILE"
fi

# Run all symbolic regression algorithms with full library
bash "$PROJECT_ROOT/Scripts/run_all_sr.sh" "$DATASET_PATH" "$NOISE_LEVEL"

echo "=========================================="
echo "Control library evaluation completed!"
echo "Results saved to: $RESULTS_FILE"
echo "==========================================" 