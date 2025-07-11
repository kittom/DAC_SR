#!/bin/bash

# KAN Activation and Execution Script
# This script activates the kan conda environment and runs the KAN symbolic regression

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

# Navigate to the KAN directory
cd "$SCRIPT_DIR/../SR_algorithms/pykan"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating KAN conda environment..."
conda activate kan

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'kan'!"
    echo "Please ensure the 'kan' conda environment exists."
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"

# Create a temporary Python script that will run KAN and extract the equation
TEMP_SCRIPT="temp_kan_analysis.py"

cat > "$TEMP_SCRIPT" << 'EOF'
import pandas as pd
import numpy as np
import torch
import sys
import os
from kan import *
from kan.utils import create_dataset, ex_round

def run_kan_analysis(csv_file_path):
    # Set up device and data type
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset
    print(f"Loading dataset from: {csv_file_path}")
    data = pd.read_csv(csv_file_path, header=None)
    data.columns = ['n', 'k', 'leading_ones']
    
    print(f"Dataset shape: {data.shape}")
    
    # Convert to torch tensors directly without normalization
    X = torch.tensor(data[['n', 'k']].values, dtype=torch.float64, device=device)
    y = torch.tensor(data['leading_ones'].values, dtype=torch.float64, device=device).reshape(-1, 1)
    
    # Create dataset dictionary
    dataset = {
        'train_input': X,
        'train_label': y,
        'test_input': X,
        'test_label': y
    }
    
    # Initialize KAN model
    print("Initializing KAN model...")
    model = KAN(width=[2, 5, 1], grid=5, k=3, seed=42, device=device)
    
    # Train the model with sparsity regularization
    print("Training KAN model...")
    model.fit(dataset, opt="LBFGS", steps=100, lamb=0.001)
    
    # Prune the model
    print("Pruning model...")
    model = model.prune()
    
    # Continue training after pruning
    print("Continuing training after pruning...")
    model.fit(dataset, opt="LBFGS", steps=50)
    
    # Refine the model with finer grid
    print("Refining model with finer grid...")
    model = model.refine(10)
    model.fit(dataset, opt="LBFGS", steps=50)
    
    # Try to find symbolic expressions automatically
    print("Attempting symbolic regression...")
    lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'abs', '1/x']
    model.auto_symbolic(lib=lib)
    
    # Extract the symbolic formula
    print("Extracting symbolic formula...")
    try:
        symbolic_formula = model.symbolic_formula()
        equation = str(ex_round(symbolic_formula[0][0], 4))
        print(f"Symbolic formula found: {equation}")
        return equation
    except Exception as e:
        print(f"Could not extract symbolic formula: {e}")
        return "ERROR: Could not extract equation"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python temp_kan_analysis.py <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    equation = run_kan_analysis(csv_file)
    
    # Get the directory of the input CSV file
    csv_dir = os.path.dirname(csv_file)
    results_file = os.path.join(csv_dir, "results.csv")
    
    # Update or create results.csv
    if os.path.exists(results_file):
        # Read existing results
        results_df = pd.read_csv(results_file)
        
        # Add or update 'kan' column
        if 'kan' not in results_df.columns:
            results_df['kan'] = ''
        
        # Update the first row with the equation
        if len(results_df) > 0:
            results_df.loc[0, 'kan'] = equation
        else:
            # If file is empty, add a row
            new_row = pd.DataFrame({'kan': [equation]})
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    else:
        # Create new results file
        results_df = pd.DataFrame({'kan': [equation]})
    
    # Save the updated results
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    print(f"Equation: {equation}")
EOF

# Run the KAN analysis
echo "Running KAN symbolic regression on CSV data..."
python "$TEMP_SCRIPT" "$CSV_FILE"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "KAN execution completed successfully!"
    
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
    echo "Error: KAN execution failed!"
    exit 1
fi

# Clean up temporary script
rm -f "$TEMP_SCRIPT"

# Deactivate the conda environment when done
echo "Deactivating conda environment..."
conda deactivate

echo "KAN execution completed!" 