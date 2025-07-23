#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/Datasets"

echo "Starting Experiment 1 Evaluation"
echo "================================="

# Function to run evaluation on a dataset
run_evaluation() {
    local dataset_path="$1"
    local dataset_name="$2"
    
    echo ""
    echo "Evaluating $dataset_name..."
    echo "Dataset: $dataset_path"
    echo "----------------------------------------"
    
    # Run all symbolic regression algorithms
    bash "$SCRIPT_DIR/../../Scripts/run_all_sr.sh" "$dataset_path"
    
    echo "Completed evaluation for $dataset_name"
}

# Evaluate OneMax dataset
if [ -f "$DATASETS_DIR/OneMax/GTOneMax.csv" ]; then
    run_evaluation "$DATASETS_DIR/OneMax/GTOneMax.csv" "OneMax"
else
    echo "Warning: OneMax dataset not found at $DATASETS_DIR/OneMax/GTOneMax.csv"
fi

# Evaluate LeadingOnes dataset
if [ -f "$DATASETS_DIR/LeadingOnes/GTLeadingOnes.csv" ]; then
    run_evaluation "$DATASETS_DIR/LeadingOnes/GTLeadingOnes.csv" "LeadingOnes"
else
    echo "Warning: LeadingOnes dataset not found at $DATASETS_DIR/LeadingOnes/GTLeadingOnes.csv"
fi

# Evaluate PSA-CMA-ES individual benchmarks
PSA_BENCHMARKS=("sphere" "ellipsoid" "rastrigin" "noisy_ellipsoid" "schaffer" "noisy_rastrigin")

for benchmark in "${PSA_BENCHMARKS[@]}"; do
    benchmark_path="$DATASETS_DIR/PSACMAES/$benchmark/psa_vars.csv"
    if [ -f "$benchmark_path" ]; then
        run_evaluation "$benchmark_path" "PSA-CMA-ES ($benchmark)"
    else
        echo "Warning: PSA-CMA-ES $benchmark dataset not found at $benchmark_path"
    fi
done

# Evaluate PSA-CMA-ES aggregated dataset
if [ -f "$DATASETS_DIR/PSACMAES/all_benchmarks.csv" ]; then
    run_evaluation "$DATASETS_DIR/PSACMAES/all_benchmarks.csv" "PSA-CMA-ES (All Benchmarks)"
else
    echo "Warning: PSA-CMA-ES aggregated dataset not found at $DATASETS_DIR/PSACMAES/all_benchmarks.csv"
fi

echo ""
echo "================================="
echo "Experiment 1 Evaluation Complete!"
echo "=================================" 