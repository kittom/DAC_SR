#!/bin/bash
set -e

# OneMax Data Generation Script
# This script generates OneMax datasets with various portfolio sizes (continuous only)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/DataGeneration/Generators/OneMax/OneMaxGT.py"

# Get output directory from command line argument
if [ $# -eq 0 ]; then
    echo "Error: No output directory provided!"
    echo "Usage: $0 <output_directory>"
    exit 1
fi

OUTPUT_DIR="$1"

echo "=========================================="
echo "OneMax Data Generation (Continuous Only)"
echo "=========================================="

# Create datasets directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/OneMax/continuous"

# Generate continuous OneMax datasets
echo "Generating continuous OneMax datasets for all portfolio sizes..."
python3 "$PYTHON_SCRIPT" 10 20 30 40 50 100 200 500 --data-type continuous --output-dir "$OUTPUT_DIR/OneMax/continuous"

echo "=========================================="
echo "OneMax data generation completed!"
echo "Continuous datasets saved to: $OUTPUT_DIR/OneMax/continuous"
echo "==========================================" 