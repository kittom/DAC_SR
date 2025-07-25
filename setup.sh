#!/bin/bash

# Automated setup for DAC_SR conda environments using environment.yml files and mamba for speed
# This script assumes conda is already installed and accessible

set -e  # Exit on any error

ENV_DIR="setup_environments"
ENVS=(analysis dso_env dso_env_rounding e2e_transformer evaluation generation kan linear psa_cmaes pysr_env q_lat tpsr)

# Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available. Please install conda first."
    exit 1
fi

# Ensure conda activate works in this shell
source ~/miniconda3/etc/profile.d/conda.sh || source $(conda info --base)/etc/profile.d/conda.sh

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