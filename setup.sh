#!/bin/bash

# Automated setup for DAC_SR conda environments using environment.yml files and mamba for speed
# This script will install miniconda if not present and create all required environments

set -e  # Exit on any error

ENV_DIR="setup_environments"
ENVS=(analysis dso_env dso_env_rounding e2e_transformer evaluation generation kan linear psa_cmaes pysr_env q_lat tpsr)
MINICONDA_INSTALLER="miniconda_installer.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Function to install miniconda
install_miniconda() {
    echo "Miniconda not found. Installing miniconda..."
    
    # Download miniconda installer
    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        echo "Downloading miniconda installer..."
        wget "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
    fi
    
    # Install miniconda
    echo "Installing miniconda to $HOME/miniconda3..."
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # Initialize conda for bash
    echo "Initializing conda..."
    "$HOME/miniconda3/bin/conda" init bash
    
    echo "Miniconda installation complete!"
    echo "Please restart your shell or run: source ~/.bashrc"
}

# Function to source conda
source_conda() {
    # Try multiple ways to source conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not find conda.sh to source"
        exit 1
    fi
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Checking for miniconda installation..."
    
    # Check if miniconda is installed but not in PATH
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        echo "Found miniconda installation at $HOME/miniconda3"
        export PATH="$HOME/miniconda3/bin:$PATH"
    else
        echo "No conda installation found. Installing miniconda..."
        install_miniconda
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi

# Verify conda is now available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is still not available after installation attempt."
    echo "Please restart your shell and run this script again."
    exit 1
fi

echo "Conda is available: $(conda --version)"

# Source conda to ensure it works properly
source_conda

# Install mamba if not already installed
if ! conda list | grep -q "^mamba[[:space:]]"; then
    echo "Mamba not found. Installing mamba for faster environment solving..."
    conda install mamba -c conda-forge -y
else
    echo "Mamba is already installed."
fi

for env in "${ENVS[@]}"; do
    YML_FILE="$ENV_DIR/${env}_environment.yml"
    if [ ! -f "$YML_FILE" ]; then
        echo "WARNING: $YML_FILE not found, skipping $env."
        continue
    fi
    # Check if environment already exists
    if conda env list | grep -q "^$env[[:space:]]"; then
        echo "Environment $env already exists, skipping."
        continue
    fi
    echo "\n========================================="
    echo "Creating conda environment: $env"
    echo "Using YAML: $YML_FILE"
    echo "========================================="
    mamba env create -f "$YML_FILE"
    echo "Environment $env setup complete."
done

echo "\nAll conda environments created successfully!"
echo "\nAvailable environments:"
conda env list

echo "\nTo activate an environment, use:"
for env in "${ENVS[@]}"; do
    echo "  conda activate $env"
done

echo "\nSetup complete!" 