#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/Datasets"
UTILS_DIR="$SCRIPT_DIR/../script_utils/evaluation"

echo "Starting Experiment 1 Evaluation"
echo "================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                    Run all evaluation types (default)"
    echo "  --control                Run control library evaluation only"
    echo "  --tailored               Run tailored library evaluation only"
    echo "  --rounding               Run rounding evaluation only"
    echo "  --datasets DATASETS      Specify datasets to evaluate (comma-separated)"
    echo "                           Options: onemax,leadingones,psacmaes,all"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run all evaluations on all datasets"
    echo "  $0 --control             # Run only control library evaluation"
    echo "  $0 --datasets onemax    # Run all evaluations on OneMax only"
    echo "  $0 --control --datasets onemax,leadingones  # Control evaluation on OneMax and LeadingOnes"
}

# Parse command line arguments
RUN_CONTROL=false
RUN_TAILORED=false
RUN_ROUNDING=false
EVALUATE_ONEMAX=false
EVALUATE_LEADINGONES=false
EVALUATE_PSACMAES=false

if [ $# -eq 0 ]; then
    # Default: run all evaluations on all datasets
    RUN_CONTROL=true
    RUN_TAILORED=true
    RUN_ROUNDING=true
    EVALUATE_ONEMAX=true
    EVALUATE_LEADINGONES=true
    EVALUATE_PSACMAES=true
else
    i=1
    while [ $i -le $# ]; do
        arg="${!i}"
        case $arg in
            --all)
                RUN_CONTROL=true
                RUN_TAILORED=true
                RUN_ROUNDING=true
                ;;
            --control)
                RUN_CONTROL=true
                ;;
            --tailored)
                RUN_TAILORED=true
                ;;
            --rounding)
                RUN_ROUNDING=true
                ;;
            --datasets)
                if [ $((i+1)) -le $# ]; then
                    i=$((i+1))
                    DATASETS="${!i}"
                    case $DATASETS in
                        onemax)
                            EVALUATE_ONEMAX=true
                            ;;
                        leadingones)
                            EVALUATE_LEADINGONES=true
                            ;;
                        psacmaes)
                            EVALUATE_PSACMAES=true
                            ;;
                        all)
                            EVALUATE_ONEMAX=true
                            EVALUATE_LEADINGONES=true
                            EVALUATE_PSACMAES=true
                            ;;
                        *)
                            echo "Error: Unknown dataset '$DATASETS'"
                            show_usage
                            exit 1
                            ;;
                    esac
                else
                    echo "Error: --datasets requires a value"
                    show_usage
                    exit 1
                fi
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option '$arg'"
                show_usage
                exit 1
                ;;
        esac
        i=$((i+1))
    done
fi

# Function to run evaluation on continuous dataset (for control and tailored library)
run_continuous_evaluation() {
    local dataset_path="$1"
    local dataset_name="$2"
    local problem_type="$3"
    local noise_level="${4:-1e-12}"

    echo ""
    echo "Evaluating $dataset_name (continuous data)..."
    echo "Dataset: $dataset_path"
    echo "Noise level: $noise_level"
    echo "----------------------------------------"

    # Get the directory containing the dataset
    local dataset_dir="$(dirname "$dataset_path")"
    
    # Run control library evaluation (results.csv)
    if [ "$RUN_CONTROL" = true ]; then
        echo "Running control library evaluation..."
        bash "$UTILS_DIR/evaluate_control_library.sh" "$dataset_path" "$noise_level"
    fi
    
    # Run tailored library evaluation (results_lib.csv)
    if [ "$RUN_TAILORED" = true ]; then
        echo "Running tailored library evaluation..."
        bash "$UTILS_DIR/evaluate_tailored_library.sh" "$problem_type" "$dataset_path" "$noise_level"
    fi

    echo "Completed continuous evaluation for $dataset_name"
    echo "Generated files:"
    if [ "$RUN_CONTROL" = true ]; then
        echo "  - $dataset_dir/results.csv (control library)"
    fi
    if [ "$RUN_TAILORED" = true ]; then
        echo "  - $dataset_dir/results_lib.csv (tailored library)"
    fi
}

# Function to run evaluation on discrete dataset (for rounding)
run_discrete_evaluation() {
    local dataset_path="$1"
    local dataset_name="$2"
    local noise_level="${3:-1e-12}"

    echo ""
    echo "Evaluating $dataset_name (discrete data)..."
    echo "Dataset: $dataset_path"
    echo "Noise level: $noise_level"
    echo "----------------------------------------"

    # Get the directory containing the dataset
    local dataset_dir="$(dirname "$dataset_path")"
    
    # Run rounding evaluation (results_rounding.csv)
    if [ "$RUN_ROUNDING" = true ]; then
        echo "Running rounding evaluation..."
        bash "$UTILS_DIR/evaluate_rounding.sh" "$dataset_path" "$noise_level"
    fi

    echo "Completed discrete evaluation for $dataset_name"
    echo "Generated files:"
    if [ "$RUN_ROUNDING" = true ]; then
        echo "  - $dataset_dir/results_rounding.csv (rounding)"
    fi
}

# Evaluate OneMax datasets
if [ "$EVALUATE_ONEMAX" = true ]; then
    # Continuous data for control and tailored library
    if [ -f "$DATASETS_DIR/OneMax/continuous/GTOneMax.csv" ]; then
        run_continuous_evaluation "$DATASETS_DIR/OneMax/continuous/GTOneMax.csv" "OneMax" "one_max"
    else
        echo "Warning: OneMax continuous dataset not found at $DATASETS_DIR/OneMax/continuous/GTOneMax.csv"
    fi
    
    # Discrete data for rounding
    if [ -f "$DATASETS_DIR/OneMax/discrete/GTOneMax.csv" ]; then
        run_discrete_evaluation "$DATASETS_DIR/OneMax/discrete/GTOneMax.csv" "OneMax"
    else
        echo "Warning: OneMax discrete dataset not found at $DATASETS_DIR/OneMax/discrete/GTOneMax.csv"
    fi
fi

# Evaluate LeadingOnes datasets
if [ "$EVALUATE_LEADINGONES" = true ]; then
    # Continuous data for control and tailored library
    if [ -f "$DATASETS_DIR/LeadingOnes/continuous/GTLeadingOnes.csv" ]; then
        run_continuous_evaluation "$DATASETS_DIR/LeadingOnes/continuous/GTLeadingOnes.csv" "LeadingOnes" "leading_ones"
    else
        echo "Warning: LeadingOnes continuous dataset not found at $DATASETS_DIR/LeadingOnes/continuous/GTLeadingOnes.csv"
    fi
    
    # Discrete data for rounding
    if [ -f "$DATASETS_DIR/LeadingOnes/discrete/GTLeadingOnes.csv" ]; then
        run_discrete_evaluation "$DATASETS_DIR/LeadingOnes/discrete/GTLeadingOnes.csv" "LeadingOnes"
    else
        echo "Warning: LeadingOnes discrete dataset not found at $DATASETS_DIR/LeadingOnes/discrete/GTLeadingOnes.csv"
    fi
fi

# Evaluate PSA-CMA-ES individual benchmarks
if [ "$EVALUATE_PSACMAES" = true ]; then
    PSA_BENCHMARKS=("sphere" "ellipsoid" "rastrigin" "noisy_ellipsoid" "schaffer" "noisy_rastrigin")

    for benchmark in "${PSA_BENCHMARKS[@]}"; do
        # Continuous data for control and tailored library
        continuous_path="$DATASETS_DIR/PSACMAES/continuous/$benchmark/psa_vars.csv"
        if [ -f "$continuous_path" ]; then
            # For noisy benchmarks, use a higher noise threshold
            if [[ "$benchmark" == *"noisy"* ]]; then
                run_continuous_evaluation "$continuous_path" "PSA-CMA-ES ($benchmark)" "psa" "0.1"
            else
                run_continuous_evaluation "$continuous_path" "PSA-CMA-ES ($benchmark)" "psa"
            fi
        else
            echo "Warning: PSA-CMA-ES $benchmark continuous dataset not found at $continuous_path"
        fi
        
        # Discrete data for rounding
        discrete_path="$DATASETS_DIR/PSACMAES/discrete/$benchmark/psa_vars.csv"
        if [ -f "$discrete_path" ]; then
            # For noisy benchmarks, use a higher noise threshold
            if [[ "$benchmark" == *"noisy"* ]]; then
                run_discrete_evaluation "$discrete_path" "PSA-CMA-ES ($benchmark)" "0.1"
            else
                run_discrete_evaluation "$discrete_path" "PSA-CMA-ES ($benchmark)"
            fi
        else
            echo "Warning: PSA-CMA-ES $benchmark discrete dataset not found at $discrete_path"
        fi
    done

    # Evaluate PSA-CMA-ES aggregated datasets
    # Continuous aggregated dataset
    if [ -f "$DATASETS_DIR/PSACMAES/continuous/all_benchmarks.csv" ]; then
        run_continuous_evaluation "$DATASETS_DIR/PSACMAES/continuous/all_benchmarks.csv" "PSA-CMA-ES (All Benchmarks)" "psa" "0.1"
    else
        echo "Warning: PSA-CMA-ES continuous aggregated dataset not found at $DATASETS_DIR/PSACMAES/continuous/all_benchmarks.csv"
    fi
    
    # Discrete aggregated dataset
    if [ -f "$DATASETS_DIR/PSACMAES/discrete/all_benchmarks.csv" ]; then
        run_discrete_evaluation "$DATASETS_DIR/PSACMAES/discrete/all_benchmarks.csv" "PSA-CMA-ES (All Benchmarks)" "0.1"
    else
        echo "Warning: PSA-CMA-ES discrete aggregated dataset not found at $DATASETS_DIR/PSACMAES/discrete/all_benchmarks.csv"
    fi
fi

echo ""
echo "================================="
echo "Experiment 1 Evaluation Complete!"
echo "================================="
echo ""
echo "Evaluation types run:"
if [ "$RUN_CONTROL" = true ]; then
    echo "- Control library (full mathematical function set) - on continuous data"
fi
if [ "$RUN_TAILORED" = true ]; then
    echo "- Tailored library (minimal function set) - on continuous data"
fi
if [ "$RUN_ROUNDING" = true ]; then
    echo "- Rounding-enabled evaluation - on discrete data"
fi
echo ""
echo "Datasets evaluated:"
if [ "$EVALUATE_ONEMAX" = true ]; then
    echo "- OneMax (continuous and discrete)"
fi
if [ "$EVALUATE_LEADINGONES" = true ]; then
    echo "- LeadingOnes (continuous and discrete)"
fi
if [ "$EVALUATE_PSACMAES" = true ]; then
    echo "- PSA-CMA-ES (continuous and discrete)"
fi
echo ""
echo "Compare the generated results files to analyze the impact of library selection and rounding." 