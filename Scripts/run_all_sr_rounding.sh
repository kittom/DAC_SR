#!/bin/bash

# Run symbolic regression algorithms with rounding support
# Usage: run_all_sr_rounding.sh <path_to_csv_file> [noise]

if [ $# -eq 0 ]; then
    echo "Error: No CSV file provided!"
    echo "Usage: $0 <path_to_csv_file> [noise]"
    exit 1
fi

CSV_FILE="$1"
NOISE="${2:-0}"

# Convert to absolute path if it's relative
if [[ ! "$CSV_FILE" = /* ]]; then
    CSV_FILE="$(pwd)/$CSV_FILE"
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run DeepSR (rounding)
echo "\n--- Running DeepSR (rounding) ---"
bash "$SCRIPTS_DIR/rounded/deepsr.sh" "$CSV_FILE" "$NOISE"
if [ $? -ne 0 ]; then
    echo "Error: DeepSR (rounding) failed! Exiting."
    exit 1
fi

# Run PySR (rounding)
echo "\n--- Running PySR (rounding) ---"
bash "$SCRIPTS_DIR/rounded/pysr.sh" "$CSV_FILE"
if [ $? -ne 0 ]; then
    echo "Error: PySR (rounding) failed! Exiting."
    exit 1
fi

# Run E2E Transformer (rounding)
echo "\n--- Running E2E Transformer (rounding) ---"
bash "$SCRIPTS_DIR/rounded/e2e_transformer.sh" "$CSV_FILE"
if [ $? -ne 0 ]; then
    echo "Error: E2E Transformer (rounding) failed! Exiting."
    exit 1
fi


echo "\nAll (rounding-enabled) symbolic regression algorithms completed." 