#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/../../DataGeneration/Generators/LeadingOnes/LeadingOnesGT.py" 10 20 30 40 50 100 200 500 --output-dir "$SCRIPT_DIR/Datasets/LeadingOnes" 