#!/bin/bash

# E2E Transformer Activation and Execution Script
# This script activates the e2e_transformer conda environment and runs the E2E Transformer symbolic regression

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

# Navigate to the E2E Transformer directory
cd "$SCRIPT_DIR/../SR_algorithms/E2E_Transformer"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating E2E Transformer conda environment..."
conda activate e2e_transformer

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'e2e_transformer'!"
    echo "Please ensure the 'e2e_transformer' conda environment exists."
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"

# Check if required files exist
if [ ! -f "model1.pt" ]; then
    echo "Error: model1.pt not found in $(pwd)"
    exit 1
fi

if [ ! -f "analyze_leading_ones.py" ]; then
    echo "Error: analyze_leading_ones.py not found in $(pwd)"
    exit 1
fi

# Run the E2E Transformer analysis
echo "Running E2E Transformer symbolic regression on CSV data..."
python analyze_leading_ones.py "$CSV_FILE"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "E2E Transformer execution completed successfully!"
    
    # Get the directory of the input CSV file
    CSV_DIR="$(dirname "$CSV_FILE")"
    RESULTS_FILE="$CSV_DIR/results.csv"
    
    # Extract the equation from the output
    EQUATION=$(python analyze_leading_ones.py "$CSV_FILE" 2>/dev/null | grep "EQUATION:" | sed 's/EQUATION: //')
    
    if [ -n "$EQUATION" ]; then
        echo "Extracted equation: $EQUATION"
        
        # Update or create results.csv
        if [ -f "$RESULTS_FILE" ]; then
            # Read existing results
            python3 -c "
import pandas as pd
import sys

try:
    results_df = pd.read_csv('$RESULTS_FILE')
    
    # Add or update 'e2e_transformer' column
    if 'e2e_transformer' not in results_df.columns:
        results_df['e2e_transformer'] = ''
    
    # Update the first row with the equation
    if len(results_df) > 0:
        results_df.loc[0, 'e2e_transformer'] = '$EQUATION'
    else:
        # If file is empty, add a row
        new_row = pd.DataFrame({'e2e_transformer': ['$EQUATION']})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Save the updated results
    results_df.to_csv('$RESULTS_FILE', index=False)
    print(f'Results updated in: $RESULTS_FILE')
    
except Exception as e:
    print(f'Error updating results: {e}')
    sys.exit(1)
"
        else
            # Create new results file
            python3 -c "
import pandas as pd

try:
    results_df = pd.DataFrame({'e2e_transformer': ['$EQUATION']})
    results_df.to_csv('$RESULTS_FILE', index=False)
    print(f'Results file created: $RESULTS_FILE')
except Exception as e:
    print(f'Error creating results file: {e}')
"
        fi
        
        if [ -f "$RESULTS_FILE" ]; then
            echo "Results saved to: $RESULTS_FILE"
            echo "Results content:"
            cat "$RESULTS_FILE"
        else
            echo "Warning: Results file not found at expected location: $RESULTS_FILE"
        fi
    else
        echo "Warning: Could not extract equation from output"
    fi
else
    echo "Error: E2E Transformer execution failed!"
    exit 1
fi

# Deactivate the conda environment when done
echo "Deactivating conda environment..."
conda deactivate

echo "E2E Transformer execution completed!" 