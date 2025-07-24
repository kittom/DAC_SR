#!/bin/bash
set -e

# Master script to run all evaluation types for all datasets
# This script provides a comprehensive way to run all three evaluation approaches

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Experiment 1: Master Evaluation Script"
echo "=========================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                    Run all evaluation types on all datasets (default)"
    echo "  --control-only           Run only control library evaluation"
    echo "  --tailored-only          Run only tailored library evaluation"
    echo "  --rounding-only          Run only rounding evaluation"
    echo "  --datasets DATASETS      Specify datasets to evaluate"
    echo "                           Options: onemax,leadingones,psacmaes,all"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run all evaluations on all datasets"
    echo "  $0 --control-only        # Run only control library evaluation"
    echo "  $0 --datasets onemax    # Run all evaluations on OneMax only"
}

# Parse command line arguments
EVAL_TYPE="all"
DATASETS="all"

if [ $# -eq 0 ]; then
    # Default: run all evaluations on all datasets
    EVAL_TYPE="all"
    DATASETS="all"
else
    for i in "$@"; do
        case $i in
            --all)
                EVAL_TYPE="all"
                ;;
            --control-only)
                EVAL_TYPE="control"
                ;;
            --tailored-only)
                EVAL_TYPE="tailored"
                ;;
            --rounding-only)
                EVAL_TYPE="rounding"
                ;;
            --datasets)
                shift
                DATASETS="$1"
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option '$i'"
                show_usage
                exit 1
                ;;
        esac
    done
fi

echo "Evaluation Type: $EVAL_TYPE"
echo "Datasets: $DATASETS"
echo "=========================================="

# Build the command for run_evaluation.sh
EVAL_CMD="bash $SCRIPT_DIR/run_evaluation.sh"

case $EVAL_TYPE in
    "all")
        EVAL_CMD="$EVAL_CMD --all"
        ;;
    "control")
        EVAL_CMD="$EVAL_CMD --control"
        ;;
    "tailored")
        EVAL_CMD="$EVAL_CMD --tailored"
        ;;
    "rounding")
        EVAL_CMD="$EVAL_CMD --rounding"
        ;;
esac

if [ "$DATASETS" != "all" ]; then
    EVAL_CMD="$EVAL_CMD --datasets $DATASETS"
fi

echo "Running command: $EVAL_CMD"
echo "=========================================="

# Execute the evaluation
eval $EVAL_CMD

echo ""
echo "=========================================="
echo "Master evaluation completed!"
echo "==========================================" 