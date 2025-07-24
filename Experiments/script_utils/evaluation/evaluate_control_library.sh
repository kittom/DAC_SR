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
RESULTS_FILE="$DATASET_DIR/results.csv"

# Check if results file exists (should be created by data generation)
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Warning: Results file not found at $RESULTS_FILE"
    echo "Please run data generation first."
    exit 1
fi

echo "Using existing results file: $RESULTS_FILE"

# Run all symbolic regression algorithms with full library
bash "$PROJECT_ROOT/Scripts/run_all_sr.sh" "$DATASET_PATH" "$NOISE_LEVEL"

echo "=========================================="
echo "Control library evaluation completed!"
echo "Results saved to: $RESULTS_FILE"
echo "==========================================" 