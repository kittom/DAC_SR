#!/bin/bash

# DeepSR Activation and Execution Script
# This script activates the dso_env conda environment and runs the deep symbolic optimization

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the DeepSR directory
cd "$SCRIPT_DIR/../SR_algorithms/DeepSR/deep-symbolic-optimization"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating DeepSR conda environment..."
conda activate dso_env

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'dso_env'!"
    echo "Please ensure the 'dso_env' conda environment exists."
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "Error: config.json not found in $(pwd)"
    exit 1
fi

# Run the deep symbolic optimization
echo "Running Deep Symbolic Optimization..."
python -m dso.run config.json

# Deactivate the conda environment when done
echo "Deactivating conda environment..."
conda deactivate

echo "DeepSR execution completed!" 