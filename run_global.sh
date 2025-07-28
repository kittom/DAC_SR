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

# Count total commands (look for any command line, not just bash)
TOTAL_COMMANDS=$(grep -c "^[^#]" "$TODO_FILE" | grep -v "^$" || echo "0")
echo "Total commands to execute: $TOTAL_COMMANDS"

# Create backup of original TODO file
BACKUP_FILE="${TODO_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$TODO_FILE" "$BACKUP_FILE"
echo "Backup created: $BACKUP_FILE"
echo ""

# Function to safely remove a command from the TODO file
remove_command() {
    local cmd="$1"
    local todo_file="$2"
    
    # Create a temporary file
    local temp_file="${todo_file}.tmp.$$"
    
    # Use awk to remove the exact line (more reliable than grep)
    awk -v cmd="$cmd" '$0 != cmd' "$todo_file" > "$temp_file"
    
    # Only replace if the temp file is different (safety check)
    if [[ -f "$temp_file" ]] && [[ $(wc -l < "$temp_file") -lt $(wc -l < "$todo_file") ]]; then
        mv "$temp_file" "$todo_file"
        return 0
    else
        rm -f "$temp_file"
        return 1
    fi
}

# Check if GNU Parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for execution..."
    echo ""
    
    # Function to execute a command and mark it as completed
    execute_command() {
        local cmd="$1"
        local todo_file="$2"
        
        echo "Executing: $cmd"
        
        # Execute the command
        if eval "$cmd"; then
            echo "✓ Success: $cmd"
            # Mark this command as completed by removing it from the TODO file
            if remove_command "$cmd" "$todo_file"; then
                echo "  Command removed from TODO file"
            else
                echo "  Warning: Could not remove command from TODO file"
            fi
        else
            echo "✗ Failed: $cmd"
        fi
    }
    
    export -f execute_command
    export -f remove_command
    
    # Create a temporary file with just the commands for parallel
    grep "^[^#]" "$TODO_FILE" | grep -v "^$" > "${TODO_FILE}.parallel"
    
    # Execute commands using GNU Parallel with the function
    # Pass both the command and the TODO file path to the function
    parallel --bar --jobs "$NUM_CPUS" --tagstring "[{1}/{2}]" \
             --joblog "global_experiment_joblog.txt" \
             execute_command :::: "${TODO_FILE}.parallel" ::: "$TODO_FILE"
             
    # Clean up temporary file
    rm -f "${TODO_FILE}.parallel"
             
else
    echo "GNU Parallel not found. Using basic parallel execution..."
    echo ""
    
    # Basic parallel execution using background jobs
    COUNTER=0
    MAX_JOBS=$NUM_CPUS
    
    while IFS= read -r line; do
        if [[ $line =~ ^[^#] ]] && [[ -n "$line" ]]; then
            COUNTER=$((COUNTER + 1))
            
            # Execute the command in background and remove from TODO file if successful
            (
                if eval "$line"; then
                    echo "[$COUNTER/$TOTAL_COMMANDS] ✓ Success: $line"
                    # Remove the command from the TODO file
                    if remove_command "$line" "$TODO_FILE"; then
                        echo "  Command removed from TODO file"
                    else
                        echo "  Warning: Could not remove command from TODO file"
                    fi
                else
                    echo "[$COUNTER/$TOTAL_COMMANDS] ✗ Failed: $line"
                fi
            ) &
            
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

# Count remaining commands
REMAINING_COMMANDS=$(grep -c "^[^#]" "$TODO_FILE" 2>/dev/null | grep -v "^$" || echo "0")

if [[ "$REMAINING_COMMANDS" -eq 0 ]]; then
    echo "✓ All experiments have been executed successfully!"
    echo "Check individual experiment directories for results."
else
    echo "⚠ $REMAINING_COMMANDS commands remaining in $TODO_FILE"
    echo "You can resume by running this script again with the same TODO file."
    echo "Remaining commands:"
    grep "^[^#]" "$TODO_FILE" | grep -v "^$" | head -5
    if [[ "$REMAINING_COMMANDS" -gt 5 ]]; then
        echo "... and $((REMAINING_COMMANDS - 5)) more"
    fi
fi 