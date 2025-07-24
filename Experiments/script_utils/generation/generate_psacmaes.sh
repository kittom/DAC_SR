#!/bin/bash
set -e

# PSA-CMA-ES Data Generation Script
# This script generates PSA-CMA-ES datasets for all benchmarks (continuous and discrete)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/DataGeneration/Generators/PSA_CMA_ES/generate_ground_truth.py"

# Get output directory from command line argument
if [ $# -eq 0 ]; then
    echo "Error: No output directory provided!"
    echo "Usage: $0 <output_directory>"
    exit 1
fi

OUTPUT_DIR="$1"

echo "=========================================="
echo "PSA-CMA-ES Data Generation"
echo "=========================================="

# Create datasets directories if they don't exist
mkdir -p "$OUTPUT_DIR/PSACMAES/continuous"
mkdir -p "$OUTPUT_DIR/PSACMAES/discrete"

# Generate continuous PSA-CMA-ES datasets
echo "Generating continuous PSA-CMA-ES datasets for all benchmarks..."
python3 "$PYTHON_SCRIPT" --iterations 1000 --data-type continuous --output-root "$OUTPUT_DIR/PSACMAES/continuous"

# Generate discrete PSA-CMA-ES datasets
echo "Generating discrete PSA-CMA-ES datasets for all benchmarks..."
python3 "$PYTHON_SCRIPT" --iterations 1000 --data-type discrete --output-root "$OUTPUT_DIR/PSACMAES/discrete"

echo "=========================================="
echo "PSA-CMA-ES data generation completed!"
echo "Continuous datasets saved to: $OUTPUT_DIR/PSACMAES/continuous"
echo "Discrete datasets saved to: $OUTPUT_DIR/PSACMAES/discrete"
echo "==========================================" 