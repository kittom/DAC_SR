#!/bin/bash

# Q-Lattice Activation and Execution Script
# This script activates the qlat conda environment and runs the Q-Lattice symbolic regression

# Check if CSV file parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: No CSV file provided!"
    echo "Usage: $0 <path_to_csv_file>"
    echo "Example: $0 ../../DataSets/Ground_Truth/LeadingOnes/continuous/GTLeadingOnes.csv"
    exit 1
fi

CSV_FILE="$1"

# Convert to absolute path if it's relative
if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

# Check if the CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the Q-Lattice directory
cd "$SCRIPT_DIR/../SR_algorithms/Q_Lattice/Code"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating Q-Lattice conda environment..."
conda activate qlat

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'qlat'!"
    echo "Please ensure the 'qlat' conda environment exists."
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in $(pwd)"
    exit 1
fi

# Run the Q-Lattice analysis
echo "Running Q-Lattice symbolic regression on CSV data..."
python main.py "$CSV_FILE"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "Q-Lattice execution completed successfully!"
    
    # Get the directory of the input CSV file
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
    echo "Error: Q-Lattice execution failed!"
    exit 1
fi

# Deactivate the conda environment when done
echo "Deactivating conda environment..."
conda deactivate

echo "Q-Lattice execution completed!" 