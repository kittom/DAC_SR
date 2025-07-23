#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/../../DataGeneration/Generators/PSA_CMA_ES/generate_ground_truth.py" --iterations 1000 --output-root "$SCRIPT_DIR/Datasets/PSACMAES" 