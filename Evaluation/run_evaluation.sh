#!/bin/bash

# Evaluation script for symbolic regression results
# Usage: run_evaluation.sh <path_to_results_file>

if [ $# -eq 0 ]; then
    echo "Error: No results file provided!"
    echo "Usage: $0 <path_to_results_file>"
    echo "Examples:"
    echo "  $0 DataSets/Ground_Truth/LeadingOnes/continuous/results.csv"
    echo "  $0 DataSets/Ground_Truth/LeadingOnes/discrete/results_rounding.csv"
    exit 1
fi

RESULTS_FILE="$1"

# Convert to absolute path if it's relative
if [[ ! "$RESULTS_FILE" = /* ]]; then
    RESULTS_FILE="$(pwd)/$RESULTS_FILE"
fi

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file '$RESULTS_FILE' not found!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running evaluation on: $RESULTS_FILE"
echo "==============================="

# Run the evaluation using the conda environment
~/miniconda3/bin/conda run -n evaluation python "$SCRIPT_DIR/evaluate_results.py" "$RESULTS_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    
    # Determine output directory
    RESULTS_DIR="$(dirname "$RESULTS_FILE")"
    if [[ "$RESULTS_FILE" == *"rounding"* ]]; then
        OUTPUT_DIR="$RESULTS_DIR/rounded_results"
    else
        OUTPUT_DIR="$RESULTS_DIR/results"
    fi
    
    echo "Results saved in: $OUTPUT_DIR"
    
    # List generated files
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Generated files:"
        ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No PNG files found"
    fi
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi 