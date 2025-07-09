#!/bin/bash

# AI-Feynman Activation and Execution Script
# This script activates the AI-Feynman conda environment and runs the main.py code

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the AI-Feynman directory
cd "$SCRIPT_DIR/../SR_algorithms/Ai_feynman"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating AI-Feynman conda environment..."
conda activate feynman

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'feynman'!"
    echo "Please ensure the 'feynman' conda environment exists."
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in $(pwd)"
    exit 1
fi

# Run the main.py script
echo "Running AI-Feynman main.py..."
python main.py

# Deactivate the conda environment when done
echo "Deactivating conda environment..."
conda deactivate

echo "AI-Feynman execution completed!"
