#!/bin/bash

# Get the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Initialize conda for this shell session
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generation

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'generation'!"
    exit 1
fi

# Parse command line arguments
INSTANCE_SIZES=()
NOISE_LEVEL="0.1"
NOISE_TYPE="gaussian"
GENERATE_NOISE=false
DATA_TYPE="continuous"
PROBLEM_TYPE="LeadingOnes"  # Default to LeadingOnes
ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[$i]}" in
        --noise-level)
            NOISE_LEVEL="${ARGS[$((i+1))]}"
            GENERATE_NOISE=true
            i=$((i+2))
            ;;
        --noise-type)
            NOISE_TYPE="${ARGS[$((i+1))]}"
            GENERATE_NOISE=true
            i=$((i+2))
            ;;
        --data-type)
            DATA_TYPE="${ARGS[$((i+1))]}"
            i=$((i+2))
            ;;
        --problem-type)
            PROBLEM_TYPE="${ARGS[$((i+1))]}"
            if [[ "$PROBLEM_TYPE" != "LeadingOnes" && "$PROBLEM_TYPE" != "OneMax" ]]; then
                echo "Error: problem-type must be 'LeadingOnes' or 'OneMax'"
                exit 1
            fi
            i=$((i+2))
            ;;
        *)
            if [[ "${ARGS[$i]}" =~ ^[0-9]+$ ]]; then
                INSTANCE_SIZES+=("${ARGS[$i]}")
            fi
            i=$((i+1))
            ;;
    esac
done

if [ ${#INSTANCE_SIZES[@]} -eq 0 ]; then
    echo "Error: No valid instance sizes provided!"
    exit 1
fi

# Set up directories based on problem type
GENERATOR_DIR="$PROJECT_ROOT/DataGeneration/Generators/$PROBLEM_TYPE"
GT_DIR="$PROJECT_ROOT/DataSets/Ground_Truth/$PROBLEM_TYPE"
GT_NOISE_DIR="$PROJECT_ROOT/DataSets/Ground_Truth_Noise/$PROBLEM_TYPE"

# Set up file names based on problem type
if [ "$PROBLEM_TYPE" = "LeadingOnes" ]; then
    CLEAN_SCRIPT="LeadingOnesGT.py"
    NOISE_SCRIPT="LeadingOnesGT_Noise.py"
    CLEAN_FILENAME="GTLeadingOnes.csv"
    NOISE_FILENAME="GTLeadingOnes_Noise_${NOISE_TYPE}_${NOISE_LEVEL}.csv"
else  # OneMax
    CLEAN_SCRIPT="OneMaxGT.py"
    NOISE_SCRIPT="OneMaxGT_Noise.py"
    CLEAN_FILENAME="GTOneMax.csv"
    NOISE_FILENAME="GTOneMax_Noise_${NOISE_TYPE}_${NOISE_LEVEL}.csv"
fi

# Clear output files only
CLEAN_DIR="$GT_DIR/$DATA_TYPE"
NOISE_DIR="$GT_NOISE_DIR/$DATA_TYPE"
CLEAN_VIZ="$CLEAN_DIR/Visualisations"
NOISE_VIZ="$NOISE_DIR/Visualisations"
mkdir -p "$CLEAN_DIR" "$NOISE_DIR" "$CLEAN_VIZ" "$NOISE_VIZ"
rm -f "$CLEAN_DIR/$CLEAN_FILENAME" "$CLEAN_VIZ"/*.png
if [ "$GENERATE_NOISE" = true ]; then
    rm -f "$NOISE_DIR/$NOISE_FILENAME" "$NOISE_VIZ"/*.png
fi

echo "Generating data for $PROBLEM_TYPE problem..."
echo "Data type: $DATA_TYPE"
echo "Instance sizes: ${INSTANCE_SIZES[@]}"

# Generate clean data
python "$GENERATOR_DIR/$CLEAN_SCRIPT" "${INSTANCE_SIZES[@]}" --data-type "$DATA_TYPE"

# Generate noisy data if requested
if [ "$GENERATE_NOISE" = true ]; then
    echo "Generating noisy data with $NOISE_TYPE noise (level: $NOISE_LEVEL)..."
    python "$GENERATOR_DIR/$NOISE_SCRIPT" "${INSTANCE_SIZES[@]}" --noise-level "$NOISE_LEVEL" --noise-type "$NOISE_TYPE" --data-type "$DATA_TYPE"
fi

# Generate clean data visualizations
for size in "${INSTANCE_SIZES[@]}"; do
    python "$GENERATOR_DIR/visualise_data.py" "$size" --data-type "$DATA_TYPE"
done
python "$GENERATOR_DIR/visualise_data.py" all --data-type "$DATA_TYPE"

# Generate noisy data visualizations if noise was generated
if [ "$GENERATE_NOISE" = true ]; then
    for size in "${INSTANCE_SIZES[@]}"; do
        python "$GENERATOR_DIR/visualise_data_noise.py" "$size" "$NOISE_FILENAME" "$NOISE_TYPE" "$NOISE_LEVEL" --data-type "$DATA_TYPE"
    done
    python "$GENERATOR_DIR/visualise_data_noise.py" all "$NOISE_FILENAME" "$NOISE_TYPE" "$NOISE_LEVEL" --data-type "$DATA_TYPE"
fi

conda deactivate

echo "Data generation and visualization pipeline completed!"
echo "Problem type: $PROBLEM_TYPE"
echo "Check $CLEAN_DIR/Visualisations for clean data plots."
if [ "$GENERATE_NOISE" = true ]; then
    echo "Check $NOISE_DIR/Visualisations for noisy data plots."
fi 