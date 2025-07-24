#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/Datasets"
UTILS_DIR="$SCRIPT_DIR/script_utils/generation"

echo "=========================================="
echo "Experiment 1: Data Generation"
echo "=========================================="

# Activate the 'generation' conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generation

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'generation'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

echo ""
echo "Starting data generation for Experiment 1..."
echo "=========================================="

# Run OneMax data generation
echo "Generating OneMax datasets..."
bash "$UTILS_DIR/generate_onemax.sh" "$DATASETS_DIR"

# Run LeadingOnes data generation
echo "Generating LeadingOnes datasets..."
bash "$UTILS_DIR/generate_leadingones.sh" "$DATASETS_DIR"

# Run PSA-CMA-ES data generation
echo "Generating PSA-CMA-ES datasets..."
bash "$UTILS_DIR/generate_psacmaes.sh" "$DATASETS_DIR"

echo ""
echo "=========================================="
echo "Experiment 1 data generation completed!"
echo "All datasets saved to: $DATASETS_DIR"
echo "==========================================" 