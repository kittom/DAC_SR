#!/bin/bash

# Usage: ./run_all_onemax_analysis.sh

echo "Finding all OneMax results CSVs in Experiments/*/Datasets/OneMax/continuous/..."

find Experiments -type f \( -name 'results_lib.csv' -o -name 'results.csv' -o -name 'results_rounding.csv' \) -path '*/Datasets/OneMax/continuous/*' | while read -r csv; do
    echo "---------------------------------------------"
    echo "Analyzing: $csv"
    ./Scripts/run_onemax_analysis.sh "$csv"
    echo "---------------------------------------------"
done

echo "All OneMax analyses complete." 