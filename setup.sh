#!/bin/bash

# Setup script for DAC_SR conda environments
# This script assumes conda is already installed and accessible

set -e  # Exit on any error

echo "Setting up conda environments for DAC_SR project..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available. Please install conda first."
    exit 1
fi

echo "Conda version: $(conda --version)"

# Accept conda channel Terms of Service if needed
echo "Accepting conda channel Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create DeepSR environment (dso_env) with Python 3.7
echo "Creating dso_env environment for DeepSR (Python 3.7)..."
conda create -n dso_env python=3.7 -y

# Create E2E_Transformer environment (e2e_transformer) with Python 3.8
echo "Creating e2e_transformer environment for E2E_Transformer (Python 3.8)..."
conda create -n e2e_transformer python=3.8 -y

# Create pykan environment (kan) with Python 3.10
echo "Creating kan environment for pykan (Python 3.10)..."
conda create -n kan python=3.10 -y

# Create Q_Lattice environment (q_lat) with latest Python
echo "Creating q_lat environment for Q_Lattice (Python 3.13)..."
conda create -n q_lat python=3.13 -y

# Create generation environment for data generation and DeepRL models
echo "Creating generation environment for data generation and DeepRL models (Python 3.10)..."
conda create -n generation python=3.10 -y

# Create PySR environment (pysr_env) with Python 3.10
# PySR requires Python >=3.8, using 3.10 for compatibility
conda create -n pysr_env python=3.10 -y

echo ""
echo "All conda environments created successfully!"
echo ""
echo "Available environments:"
conda env list
echo ""
echo "To activate an environment, use:"
echo "  conda activate dso_env      # For DeepSR"
echo "  conda activate e2e_transformer  # For E2E_Transformer"
echo "  conda activate kan          # For pykan"
echo "  conda activate q_lat        # For Q_Lattice"
echo "  conda activate generation   # For data generation and DeepRL models"
echo "  conda activate pysr_env        # For PySR"
echo ""
echo "Setup complete!" 