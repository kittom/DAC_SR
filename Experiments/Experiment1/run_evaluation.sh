#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/Datasets"
UTILS_DIR="$SCRIPT_DIR/script_utils/evaluation"

echo "Starting Experiment 1 Evaluation"
echo "================================="

# Function to run evaluation on a dataset
run_evaluation() {
    local dataset_path="$1"
    local dataset_name="$2"
    local problem_type="$3"
    local noise_level="${4:-1e-12}"

    echo ""
    echo "Evaluating $dataset_name..."
    echo "Dataset: $dataset_path"
    echo "Noise level: $noise_level"
    echo "----------------------------------------"

    # Get the directory containing the dataset
    local dataset_dir="$(dirname "$dataset_path")"
    
    # Run control library evaluation (results.csv)
    echo "Running control library evaluation..."
    bash "$UTILS_DIR/evaluate_control_library.sh" "$dataset_path" "$noise_level"
    
    # Run tailored library evaluation (results_lib.csv)
    echo "Running tailored library evaluation..."
    bash "$UTILS_DIR/evaluate_tailored_library.sh" "$problem_type" "$dataset_path" "$noise_level"
    
    # Run rounding evaluation (results_rounding.csv)
    echo "Running rounding evaluation..."
    bash "$UTILS_DIR/evaluate_rounding.sh" "$dataset_path" "$noise_level"

    echo "Completed evaluation for $dataset_name"
    echo "Generated files:"
    echo "  - $dataset_dir/results.csv (control library)"
    echo "  - $dataset_dir/results_lib.csv (tailored library)"
    echo "  - $dataset_dir/results_rounding.csv (rounding)"
}

# Evaluate OneMax dataset
if [ -f "$DATASETS_DIR/OneMax/GTOneMax.csv" ]; then
    run_evaluation "$DATASETS_DIR/OneMax/GTOneMax.csv" "OneMax" "one_max"
else
    echo "Warning: OneMax dataset not found at $DATASETS_DIR/OneMax/GTOneMax.csv"
fi

# Evaluate LeadingOnes dataset
if [ -f "$DATASETS_DIR/LeadingOnes/GTLeadingOnes.csv" ]; then
    run_evaluation "$DATASETS_DIR/LeadingOnes/GTLeadingOnes.csv" "LeadingOnes" "leading_ones"
else
    echo "Warning: LeadingOnes dataset not found at $DATASETS_DIR/LeadingOnes/GTLeadingOnes.csv"
fi

# Evaluate PSA-CMA-ES individual benchmarks
PSA_BENCHMARKS=("sphere" "ellipsoid" "rastrigin" "noisy_ellipsoid" "schaffer" "noisy_rastrigin")

for benchmark in "${PSA_BENCHMARKS[@]}"; do
    benchmark_path="$DATASETS_DIR/PSACMAES/$benchmark/psa_vars.csv"
    if [ -f "$benchmark_path" ]; then
        # For noisy benchmarks, use a higher noise threshold
        if [[ "$benchmark" == *"noisy"* ]]; then
            run_evaluation "$benchmark_path" "PSA-CMA-ES ($benchmark)" "psa" "0.1"
        else
            run_evaluation "$benchmark_path" "PSA-CMA-ES ($benchmark)" "psa"
        fi
    else
        echo "Warning: PSA-CMA-ES $benchmark dataset not found at $benchmark_path"
    fi
done

# Evaluate PSA-CMA-ES aggregated dataset
if [ -f "$DATASETS_DIR/PSACMAES/all_benchmarks.csv" ]; then
    run_evaluation "$DATASETS_DIR/PSACMAES/all_benchmarks.csv" "PSA-CMA-ES (All Benchmarks)" "psa" "0.1"
else
    echo "Warning: PSA-CMA-ES aggregated dataset not found at $DATASETS_DIR/PSACMAES/all_benchmarks.csv"
fi

echo ""
echo "================================="
echo "Experiment 1 Evaluation Complete!"
echo "================================="
echo ""
echo "Three types of results generated for each dataset:"
echo "- results.csv: Control library (full mathematical function set)"
echo "- results_lib.csv: Tailored library (minimal function set)"
echo "- results_rounding.csv: Rounding-enabled evaluation"
echo ""
echo "Compare the three files to analyze the impact of library selection and rounding." 