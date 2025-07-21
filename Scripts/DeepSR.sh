#!/bin/bash

# DeepSR Activation and Execution Script
# This script activates the dso_env conda environment and runs the deep symbolic optimization

if [ $# -lt 1 ]; then
    echo "Error: No CSV file provided!"
    echo "Usage: $0 <path_to_csv_file> [noise]"
    echo "Example: $0 ../../DataSets/Ground_Truth/LeadingOnes/continuous/GTLeadingOnes.csv 0.05"
    exit 1
fi

CSV_FILE="$1"
NOISE="${2:-0}"

if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../SR_algorithms/DeepSR/deep-symbolic-optimization"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating DeepSR conda environment..."
conda activate dso_env

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'dso_env'!"
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Input CSV file: $CSV_FILE"
echo "Noise parameter: $NOISE"

# Prepare config.json from template
THRESHOLD=1e-12
REWARD_NOISE=0.0
if (( $(echo "$NOISE > 0" | bc -l) )); then
    THRESHOLD="$NOISE"
    REWARD_NOISE="$NOISE"
fi

sed -e "s|__DATASET__|$CSV_FILE|g" \
    -e "s|__THRESHOLD__|$THRESHOLD|g" \
    -e "s|__REWARD_NOISE__|$REWARD_NOISE|g" \
    config_template.json > config.json

echo "Running Deep Symbolic Regression on CSV data..."
python run.py "$CSV_FILE" --config config.json

if [ $? -eq 0 ]; then
    echo "DeepSR execution completed successfully!"
    CSV_DIR="$(dirname "$CSV_FILE")"
    RESULTS_FILE="$CSV_DIR/results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Results saved to: $RESULTS_FILE"
        echo "Results content:"
        cat "$RESULTS_FILE"
    else
        echo "Warning: Results file not found at expected location: $RESULTS_FILE"
    fi
else
    echo "Error: DeepSR execution failed!"
    exit 1
fi

echo "Deactivating conda environment..."
conda deactivate

echo "DeepSR execution completed!" 