#!/bin/bash
set -e

# LeadingOnes Data Generation Script
# This script generates LeadingOnes datasets with various portfolio sizes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/DataGeneration/Generators/LeadingOnes/LeadingOnesGT.py"

# Get output directory from command line argument
if [ $# -eq 0 ]; then
    echo "Error: No output directory provided!"
    echo "Usage: $0 <output_directory>"
    exit 1
fi

OUTPUT_DIR="$1"

echo "=========================================="
echo "LeadingOnes Data Generation"
echo "=========================================="

# Create datasets directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/LeadingOnes"

# Generate all LeadingOnes datasets at once
echo "Generating LeadingOnes datasets for all portfolio sizes..."
python3 "$PYTHON_SCRIPT" 10 20 30 40 50 100 200 500 --output-dir "$OUTPUT_DIR/LeadingOnes"

echo "=========================================="
echo "LeadingOnes data generation completed!"
echo "Datasets saved to: $OUTPUT_DIR/LeadingOnes"
echo "==========================================" 