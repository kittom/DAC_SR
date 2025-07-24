#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/Datasets"
UTILS_DIR="$SCRIPT_DIR/../script_utils/generation"

echo "=========================================="
echo "Experiment 1: Data Generation"
echo "=========================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                    Generate all datasets (default)"
    echo "  --onemax                 Generate OneMax datasets only"
    echo "  --leadingones            Generate LeadingOnes datasets only"
    echo "  --psacmaes               Generate PSA-CMA-ES datasets only"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Generate all datasets"
    echo "  $0 --onemax              # Generate OneMax only"
    echo "  $0 --leadingones --psacmaes  # Generate LeadingOnes and PSA-CMA-ES"
}

# Parse command line arguments
GENERATE_ONEMAX=false
GENERATE_LEADINGONES=false
GENERATE_PSACMAES=false

if [ $# -eq 0 ]; then
    # Default: generate all
    GENERATE_ONEMAX=true
    GENERATE_LEADINGONES=true
    GENERATE_PSACMAES=true
else
    for arg in "$@"; do
        case $arg in
            --all)
                GENERATE_ONEMAX=true
                GENERATE_LEADINGONES=true
                GENERATE_PSACMAES=true
                ;;
            --onemax)
                GENERATE_ONEMAX=true
                ;;
            --leadingones)
                GENERATE_LEADINGONES=true
                ;;
            --psacmaes)
                GENERATE_PSACMAES=true
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option '$arg'"
                show_usage
                exit 1
                ;;
        esac
    done
fi

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
if [ "$GENERATE_ONEMAX" = true ]; then
    echo "Generating OneMax datasets..."
    bash "$UTILS_DIR/generate_onemax.sh" "$DATASETS_DIR"
fi

# Run LeadingOnes data generation
if [ "$GENERATE_LEADINGONES" = true ]; then
    echo "Generating LeadingOnes datasets..."
    bash "$UTILS_DIR/generate_leadingones.sh" "$DATASETS_DIR"
fi

# Run PSA-CMA-ES data generation
if [ "$GENERATE_PSACMAES" = true ]; then
    echo "Generating PSA-CMA-ES datasets..."
    bash "$UTILS_DIR/generate_psacmaes.sh" "$DATASETS_DIR"
fi

echo ""
echo "=========================================="
echo "Experiment 1 data generation completed!"
echo "All datasets saved to: $DATASETS_DIR"
echo "==========================================" 