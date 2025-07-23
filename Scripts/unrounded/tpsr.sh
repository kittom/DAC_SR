#!/bin/bash

# TPSR Activation and Execution Script (Unrounded)
# This script activates the tpsr conda environment and runs TPSR symbolic regression (unrounded)

if [ $# -eq 0 ]; then
    echo "Error: No CSV file provided!"
    echo "Usage: $0 <path_to_csv_file>"
    echo "Example: $0 ../../DataSets/Ground_Truth/LeadingOnes/continuous/GTLeadingOnes.csv"
    exit 1
fi

CSV_FILE="$1"

if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../SR_algorithms/TPSR"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating tpsr conda environment..."
conda activate tpsr

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'tpsr'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"

echo "Running TPSR symbolic regression on CSV data..."
python run_tpsr_on_csv.py "$CSV_FILE"

if [ $? -eq 0 ]; then
    echo "TPSR execution completed successfully!"
    CSV_DIR="$(dirname "$CSV_FILE")"
    RESULTS_FILE="$CSV_DIR/results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Results saved to: $RESULTS_FILE"
        echo "Results content:"
        cat "$RESULTS_FILE"
    else
        echo "Warning: Results file not found at expected location: $RESULTS_FILE"
    fi
else
    echo "Error: TPSR execution failed!"
    exit 1
fi

echo "Deactivating conda environment..."
conda deactivate

echo "TPSR execution completed!" 