#!/bin/bash

# Setup script for Linear Regression conda environment

echo "Setting up Linear Regression conda environment..."

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create conda environment
echo "Creating 'linear' conda environment..."
conda create -n linear python=3.9 -y

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment 'linear'!"
    exit 1
fi

# Activate the environment
echo "Activating 'linear' conda environment..."
conda activate linear

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'linear'!"
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0

if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages!"
    exit 1
fi

echo "Linear Regression environment setup completed successfully!"
echo "To activate the environment, run: conda activate linear" 