#!/bin/bash

# Usage: generate_leadingones_csv.sh [rounded|unrounded]
# Default is rounded if not specified

MODE="${1:-rounded}"

# Set output directory and CLI flag
if [ "$MODE" = "unrounded" ]; then
    OUTDIR="../../../DataSets/DeepRL/LeadingOnes/UnRounded"
    FLAG="--unrounded"
else
    OUTDIR="../../../DataSets/DeepRL/LeadingOnes/Rounded"
    FLAG="--rounded"
fi

mkdir -p "$OUTDIR"
OUTPUT_PATH="$OUTDIR/LeadingOnesModel.csv"

# Activate conda environment
echo "Activating generation conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generation

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'generation'!"
    exit 1
fi

echo "Running generate_csv_ppo.py with $FLAG, output: $OUTPUT_PATH"
python generate_csv_ppo.py $FLAG

# Move the generated file to the correct output path (if not already there)
if [ -f ../../../DataSets/DeepRL/LeadingOnesModel.csv ]; then
    mv ../../../DataSets/DeepRL/LeadingOnesModel.csv "$OUTPUT_PATH"
fi

echo "CSV saved to $OUTPUT_PATH"

conda deactivate 