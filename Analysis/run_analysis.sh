#!/bin/bash

# Analysis script for symbolic regression results
# Usage: run_analysis.sh <path_to_results_file>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_results_file>"
    exit 1
fi

RESULTS_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "❌ File not found: $RESULTS_FILE"
    exit 1
fi

echo "Running analysis on: $RESULTS_FILE"

# Run the analysis using the conda environment
~/miniconda3/bin/conda run -n evaluation python "$SCRIPT_DIR/analyze_results.py" "$RESULTS_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
else
    echo "❌ Analysis failed!"
fi 