#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the 'generation' conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generation

# Run all data generation scripts using absolute paths
"$SCRIPT_DIR/generate_onemax.sh"
"$SCRIPT_DIR/generate_leadingones.sh"
"$SCRIPT_DIR/generate_psacmaes.sh" 