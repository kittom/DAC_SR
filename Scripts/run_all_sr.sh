#!/bin/bash

# Run all symbolic regression algorithms on a provided CSV file
# Usage: run_all_sr.sh <path_to_csv_file> [noise]

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

run_and_check() {
    local script_name="$1"
    shift
    echo ""
    echo "==============================="
    echo "Running $script_name on $CSV_FILE $@"
    echo "==============================="
    bash "$SCRIPTS_DIR/$script_name" "$CSV_FILE" "$@"
    if [ $? -ne 0 ]; then
        echo "Error: $script_name failed! Exiting."
        exit 1
    fi
}
run_and_check DeepSR.sh "$NOISE"
run_and_check e2e_transformer.sh
run_and_check qlattice.sh
run_and_check kan.sh
run_and_check pysr.sh

echo ""
echo "All symbolic regression algorithms completed." 