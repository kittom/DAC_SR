#!/bin/bash

# Global Experiment Runner
# This script executes all experiments from the global TODO list

# Default values
TODO_FILE="global_TODO.txt"
NUM_CPUS=$(nproc)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--todo)
            TODO_FILE="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_CPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --todo FILE    TODO file to use (default: global_TODO.txt)"
            echo "  -j, --jobs N       Number of parallel jobs (default: all available cores)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if TODO file exists
if [[ ! -f "$TODO_FILE" ]]; then
    echo "Error: TODO file '$TODO_FILE' not found!"
    echo "Make sure you've run the master experiment runner first:"
    echo "  python master_experiment_runner.py"
    exit 1
fi

# Display system information
echo "=== Global Experiment Runner ==="
echo "TODO file: $TODO_FILE"
echo "Available CPU cores: $(nproc)"
echo "Using CPU cores: $NUM_CPUS"
echo "================================"

# Count total commands
TOTAL_COMMANDS=$(grep -c "^bash " "$TODO_FILE")
echo "Total commands to execute: $TOTAL_COMMANDS"
echo ""

# Check if GNU Parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for execution..."
    echo ""
    
    # Execute commands using GNU Parallel
    parallel --bar --jobs "$NUM_CPUS" --tagstring "[{1}/{2}]" \
             --joblog "global_experiment_joblog.txt" \
             :::: <(grep "^bash " "$TODO_FILE") \
             ::: "$TOTAL_COMMANDS"
             
else
    echo "GNU Parallel not found. Using basic parallel execution..."
    echo ""
    
    # Basic parallel execution using background jobs
    COUNTER=0
    MAX_JOBS=$NUM_CPUS
    
    while IFS= read -r line; do
        if [[ $line =~ ^bash ]]; then
            COUNTER=$((COUNTER + 1))
            echo "[$COUNTER/$TOTAL_COMMANDS] Executing: $line"
            
            # Execute the command in background
            eval "$line" &
            
            # Wait if we've reached max jobs
            if [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; then
                wait -n
            fi
        fi
    done < "$TODO_FILE"
    
    # Wait for all remaining jobs
    wait
fi

echo ""
echo "=== Global Experiment Runner Complete ==="
echo "All experiments have been executed!"
echo "Check individual experiment directories for results." 