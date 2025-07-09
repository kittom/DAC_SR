#!/bin/bash

# GenerateData.sh - Data Generation and Visualization Script
# This script activates the generation conda environment and runs the complete pipeline

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the Generators directory
cd "$SCRIPT_DIR/../DataSets/Generators"

# Initialize conda for this shell session
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating generation conda environment..."
conda activate generation

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'generation'!"
    echo "Please ensure the 'generation' conda environment exists."
    exit 1
fi

echo "Conda environment activated successfully!"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <instance_size1> [instance_size2] [instance_size3] ... [--noise-level <level>] [--noise-type <type>]"
    echo "Example: $0 10 20 30"
    echo "Example: $0 50 100 --noise-level 0.2 --noise-type gaussian"
    echo "Example: $0 10 20 30 40 50 --noise-level 0.1 --noise-type uniform"
    exit 1
fi

# Parse command line arguments
INSTANCE_SIZES=()
NOISE_LEVEL="0.1"
NOISE_TYPE="gaussian"
GENERATE_NOISE=false

# Convert arguments to array for easier processing
ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[$i]}" in
        --noise-level)
            if [ $((i+1)) -lt ${#ARGS[@]} ]; then
                NOISE_LEVEL="${ARGS[$((i+1))]}"
                GENERATE_NOISE=true
                i=$((i+2))
            else
                echo "Error: --noise-level requires a value"
                exit 1
            fi
            ;;
        --noise-type)
            if [ $((i+1)) -lt ${#ARGS[@]} ]; then
                NOISE_TYPE="${ARGS[$((i+1))]}"
                GENERATE_NOISE=true
                i=$((i+2))
            else
                echo "Error: --noise-type requires a value"
                exit 1
            fi
            ;;
        *)
            # Check if it's a number (instance size)
            if [[ "${ARGS[$i]}" =~ ^[0-9]+$ ]]; then
                INSTANCE_SIZES+=("${ARGS[$i]}")
            else
                echo "Warning: Ignoring non-numeric argument: ${ARGS[$i]}"
            fi
            i=$((i+1))
            ;;
    esac
done

# Check if we have instance sizes
if [ ${#INSTANCE_SIZES[@]} -eq 0 ]; then
    echo "Error: No valid instance sizes provided!"
    exit 1
fi

echo "Generating data for instance sizes: ${INSTANCE_SIZES[*]}"
if [ "$GENERATE_NOISE" = true ]; then
    echo "Noise settings: level=$NOISE_LEVEL, type=$NOISE_TYPE"
fi

# Clear output directories
echo "Clearing output directories..."
if [ -d "../Ground_Truth" ]; then
    rm -rf ../Ground_Truth/GTLeadingOnes.csv
    rm -rf ../Ground_Truth/Visualisations/
    echo "Cleared ../Ground_Truth/ directory"
fi

if [ "$GENERATE_NOISE" = true ]; then
    if [ -d "../Ground_Truth_Noise" ]; then
        rm -rf ../Ground_Truth_Noise/GTLeadingOnes_Noise_*.csv
        rm -rf ../Ground_Truth_Noise/Visualisations/
        echo "Cleared ../Ground_Truth_Noise/ directory"
    fi
fi

# Generate the clean data
echo "Step 1: Generating LeadingOnes ground truth data..."
python LeadingOnesGT.py "${INSTANCE_SIZES[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Clean data generation failed!"
    exit 1
fi

echo "Clean data generation completed successfully!"

# Generate noisy data if requested
if [ "$GENERATE_NOISE" = true ]; then
    echo "Step 2: Generating LeadingOnes ground truth data with noise..."
    python LeadingOnesGT_Noise.py "${INSTANCE_SIZES[@]}" --noise-level "$NOISE_LEVEL" --noise-type "$NOISE_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "Error: Noisy data generation failed!"
        exit 1
    fi
    
    echo "Noisy data generation completed successfully!"
fi

# Generate clean data visualizations
echo "Step 3: Generating clean data visualizations..."
for size in "${INSTANCE_SIZES[@]}"; do
    echo "Generating 2D plot for instance size: $size (clean data)"
    python visualise_data.py "$size"
    
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to generate clean visualization for instance size $size"
    fi
done

# Generate the 3D comparison plot for clean data
echo "Step 4: Generating 3D comparison plot (clean data)..."
python visualise_data.py all

if [ $? -ne 0 ]; then
    echo "Warning: Failed to generate 3D comparison plot for clean data"
fi

# Generate noisy data visualizations if noise was generated
if [ "$GENERATE_NOISE" = true ]; then
    echo "Step 5: Generating noisy data visualizations..."
    
    # Get the noise filename that was generated
    NOISE_FILENAME="GTLeadingOnes_Noise_${NOISE_TYPE}_${NOISE_LEVEL}.csv"
    
    for size in "${INSTANCE_SIZES[@]}"; do
        echo "Generating 2D plot for instance size: $size (noisy data)"
        python visualise_data_noise.py "$size" "$NOISE_FILENAME" "$NOISE_TYPE" "$NOISE_LEVEL"
        
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to generate noisy visualization for instance size $size"
        fi
    done
    
    # Generate the 3D comparison plot for noisy data
    echo "Step 6: Generating 3D comparison plot (noisy data)..."
    python visualise_data_noise.py all "$NOISE_FILENAME" "$NOISE_TYPE" "$NOISE_LEVEL"
    
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to generate 3D comparison plot for noisy data"
    fi
fi

echo "All visualizations completed!"

# Deactivate the conda environment when done
echo "Deactivating conda environment..."
conda deactivate

echo "Data generation and visualization pipeline completed!"
echo "Check the ../Ground_Truth/Visualisations/ directory for clean data plots."
if [ "$GENERATE_NOISE" = true ]; then
    echo "Check the ../Ground_Truth_Noise/Visualisations/ directory for noisy data plots."
fi 