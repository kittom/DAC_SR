#!/bin/bash

# Usage: ./run_all_leadingones_analysis.sh

echo "Finding all LeadingOnes results CSVs in Experiments/*/Datasets/LeadingOnes/continuous/..."

find Experiments -type f \( -name 'results_lib.csv' -o -name 'results.csv' -o -name 'results_rounding.csv' \) -path '*/Datasets/LeadingOnes/continuous/*' | while read -r csv; do
    echo "---------------------------------------------"
    echo "Analyzing: $csv"
    ./Scripts/run_leadingones_analysis.sh "$csv"
    echo "---------------------------------------------"
done

echo "All LeadingOnes analyses complete." 