#!/bin/bash

# Usage: ./run_onemax_analysis.sh <path_to_results_csv>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_results_csv>"
    exit 1
fi

RESULTS_CSV="$1"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "Error: File not found: $RESULTS_CSV"
    exit 1
fi

# Activate the evaluation conda environment and run the analysis
echo "Activating conda environment 'evaluation'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate evaluation

echo "Running OneMax analysis on $RESULTS_CSV..."
python3 Analysis/analyze_onemax.py "$RESULTS_CSV"

conda deactivate 