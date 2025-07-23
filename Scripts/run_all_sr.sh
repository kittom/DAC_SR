#!/bin/bash

# Script to run all symbolic regression algorithms on a given CSV file
# This script orchestrates the execution of various SR algorithms and updates results.csv

if [ $# -lt 1 ]; then
    echo "Error: No CSV file provided!"
    echo "Usage: $0 <path_to_csv_file> [noise]"
    echo "Example: $0 ../../DataSets/Ground_Truth/LeadingOnes/continuous/GTLeadingOnes.csv 0.05"
    exit 1
fi

CSV_FILE="$1"
NOISE="${2:-1e-12}"

if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Running All Symbolic Regression Algorithms"
echo "=========================================="
echo "CSV File: $CSV_FILE"
echo "Noise Parameter: $NOISE"
echo "=========================================="

# Function to run an algorithm and check for success
run_and_check() {
    local algorithm_name="$1"
    local script_path="$2"
    
    echo ""
    echo "Running $algorithm_name..."
    echo "----------------------------------------"
    
    if [ -f "$script_path" ]; then
        bash "$script_path" "$CSV_FILE" "$NOISE"
        if [ $? -eq 0 ]; then
            echo "✓ $algorithm_name completed successfully"
        else
            echo "✗ $algorithm_name failed"
        fi
    else
        echo "✗ Script not found: $script_path"
    fi
}

# Run all algorithms with noise parameter
run_and_check "DeepSR" "$SCRIPT_DIR/unrounded/deepsr.sh"
run_and_check "PySR" "$SCRIPT_DIR/unrounded/pysr.sh"
run_and_check "KAN" "$SCRIPT_DIR/unrounded/kan.sh"
run_and_check "Q-Lattice" "$SCRIPT_DIR/unrounded/qlattice.sh"
run_and_check "E2E Transformer" "$SCRIPT_DIR/unrounded/e2e_transformer.sh"
run_and_check "TPSR" "$SCRIPT_DIR/unrounded/tpsr.sh"
run_and_check "Linear Regression" "$SCRIPT_DIR/unrounded/linear.sh"

echo ""
echo "=========================================="
echo "All algorithms completed!"
echo "=========================================="

# Display final results
CSV_DIR="$(dirname "$CSV_FILE")"
RESULTS_FILE="$CSV_DIR/results.csv"

if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Final Results Summary:"
    echo "======================"
    cat "$RESULTS_FILE"
    echo ""
    echo "Results saved to: $RESULTS_FILE"
else
    echo ""
    echo "Warning: No results.csv file found at: $RESULTS_FILE"
fi

echo ""
echo "=========================================="
echo "Symbolic Regression Evaluation Complete!"
echo "==========================================" 