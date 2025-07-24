#!/bin/bash
set -e

# LeadingOnes Data Generation Script
# This script generates LeadingOnes datasets with various portfolio sizes (continuous and discrete)

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

# Create datasets directories if they don't exist
mkdir -p "$OUTPUT_DIR/LeadingOnes/continuous"
mkdir -p "$OUTPUT_DIR/LeadingOnes/discrete"

# Generate continuous LeadingOnes datasets
echo "Generating continuous LeadingOnes datasets for all portfolio sizes..."
python3 "$PYTHON_SCRIPT" 10 20 30 40 50 100 200 500 --data-type continuous --output-dir "$OUTPUT_DIR/LeadingOnes/continuous"

# Generate discrete LeadingOnes datasets
echo "Generating discrete LeadingOnes datasets for all portfolio sizes..."
python3 "$PYTHON_SCRIPT" 10 20 30 40 50 100 200 500 --data-type discrete --output-dir "$OUTPUT_DIR/LeadingOnes/discrete"

echo "=========================================="
echo "LeadingOnes data generation completed!"
echo "Continuous datasets saved to: $OUTPUT_DIR/LeadingOnes/continuous"
echo "Discrete datasets saved to: $OUTPUT_DIR/LeadingOnes/discrete"
echo "==========================================" 