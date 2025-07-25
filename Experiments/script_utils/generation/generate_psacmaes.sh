#!/bin/bash
set -e

# PSA-CMA-ES Data Generation Script
# This script generates PSA-CMA-ES datasets for all benchmarks (continuous only)

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
echo "PSA-CMA-ES Data Generation (Continuous Only)"
echo "=========================================="

# Create datasets directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/PSACMAES"

# Generate PSA-CMA-ES datasets
echo "Generating PSA-CMA-ES datasets for all benchmarks..."
python3 "$PYTHON_SCRIPT" --iterations 1000 --data-type continuous --output-root "$OUTPUT_DIR/PSACMAES"

echo "=========================================="
echo "PSA-CMA-ES data generation completed!"
echo "Datasets saved to: $OUTPUT_DIR/PSACMAES"
echo "==========================================" 