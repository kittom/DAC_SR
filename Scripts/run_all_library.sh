#!/bin/bash
set -e

# Script to run all symbolic regression algorithms with minimal library configurations
# This script orchestrates the execution of various SR algorithms with problem-specific minimal libraries

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments!"
    echo "Usage: $0 <problem_type> <path_to_csv_file> [noise]"
    echo "Example: $0 one_max ../../DataSets/Ground_Truth/OneMax/continuous/GTOneMax.csv 0.05"
    echo ""
    echo "Problem types:"
    echo "  one_max      - OneMax problem (sqrt(x1/(x1-x2)))"
    echo "  leading_ones - LeadingOnes problem (x1/(x2 + 1))"
    echo "  psa          - PSA-CMA-ES problem (x1 * exp(x2 * (x5 - (x3 / x4))))"
    echo ""
    echo "The script will run all algorithms with minimal function libraries"
    echo "and save results to results_lib.csv in the same directory as the CSV file."
    exit 1
fi

PROBLEM_TYPE="$1"
CSV_FILE="$2"
NOISE="${3:-1e-12}"

# Validate problem type
if [[ "$PROBLEM_TYPE" != "one_max" && "$PROBLEM_TYPE" != "leading_ones" && "$PROBLEM_TYPE" != "psa" ]]; then
    echo "Error: Problem type must be one of: one_max, leading_ones, psa"
    exit 1
fi

# Convert to absolute path if it's relative
if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Minimal Library Symbolic Regression Suite"
echo "=========================================="
echo "Problem Type: $PROBLEM_TYPE"
echo "CSV File: $CSV_FILE"
echo "Noise Parameter: $NOISE"
echo "=========================================="

# Function to run an algorithm and check for success
run_and_check() {
    local algorithm_name="$1"
    local script_path="$2"
    
    echo ""
    echo "Running $algorithm_name with minimal library..."
    echo "----------------------------------------"
    
    if [ -f "$script_path" ]; then
        bash "$script_path" "$CSV_FILE" "$PROBLEM_TYPE" "$NOISE"
        if [ $? -eq 0 ]; then
            echo "✓ $algorithm_name completed successfully"
        else
            echo "✗ $algorithm_name failed"
        fi
    else
        echo "✗ Script not found: $script_path"
    fi
}

# Run all algorithms with minimal library configurations
run_and_check "DeepSR" "$SCRIPT_DIR/unrounded/algorithm_library/deepsr_lib.sh"
run_and_check "PySR" "$SCRIPT_DIR/unrounded/algorithm_library/pysr_lib.sh"
run_and_check "KAN" "$SCRIPT_DIR/unrounded/algorithm_library/kan_lib.sh"
run_and_check "TPSR" "$SCRIPT_DIR/unrounded/algorithm_library/tpsr_lib.sh"
run_and_check "Linear Regression" "$SCRIPT_DIR/unrounded/algorithm_library/linear_lib.sh"

echo ""
echo "=========================================="
echo "All minimal library algorithms completed!"
echo "=========================================="

# Display final results
CSV_DIR="$(dirname "$CSV_FILE")"
RESULTS_FILE="$CSV_DIR/results_lib.csv"

if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Final Results Summary (Minimal Library):"
    echo "======================================="
    cat "$RESULTS_FILE"
    echo ""
    echo "Results saved to: $RESULTS_FILE"
else
    echo ""
    echo "Warning: No results_lib.csv file found at: $RESULTS_FILE"
fi

echo ""
echo "=========================================="
echo "Minimal Library Evaluation Complete!"
echo "=========================================="
echo ""
echo "Note: This experiment used minimal function libraries:"
case "$PROBLEM_TYPE" in
    "one_max")
        echo "  OneMax: sqrt(x1/(x1-x2)) - Required: +, -, /, sqrt"
        ;;
    "leading_ones")
        echo "  LeadingOnes: x1/(x2 + 1) - Required: +, /"
        ;;
    "psa")
        echo "  PSA-CMA-ES: x1 * exp(x2 * (x5 - (x3 / x4))) - Required: *, -, /, exp"
        ;;
esac
echo ""
echo "Compare with full library results in results.csv" 