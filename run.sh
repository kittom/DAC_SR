#!/bin/bash
# run: Parallel job dispatcher for TODO.txt
# Usage: ./run [NUM_CPUS]

set -euo pipefail

TODO_FILE="TODO.txt"
NUM_CPUS="${1:-$(nproc)}"

if ! [[ -f "$TODO_FILE" ]]; then
    echo "Error: $TODO_FILE not found in current directory."
    exit 1
fi

# Extract all non-comment, non-blank lines (actual commands)
mapfile -t JOBS < <(grep -vE '^#|^$' "$TODO_FILE")
TOTAL_JOBS="${#JOBS[@]}"

if [[ "$TOTAL_JOBS" -eq 0 ]]; then
    echo "No jobs found in $TODO_FILE."
    exit 0
fi

# Function to run a single job and print progress
run_job() {
    local job_idx="$1"
    local job_cmd="$2"
    echo "[START] Job $((job_idx+1))/$TOTAL_JOBS: $job_cmd"
    eval "$job_cmd"
    local status=$?
    if [[ $status -eq 0 ]]; then
        echo "[DONE]  Job $((job_idx+1))/$TOTAL_JOBS: $job_cmd"
    else
        echo "[FAIL]  Job $((job_idx+1))/$TOTAL_JOBS: $job_cmd (exit code $status)"
    fi
    return $status
}

# Export function for subshells
export -f run_job
export TOTAL_JOBS

# Main job dispatch loop using GNU parallel if available, else fallback to bash
if command -v parallel > /dev/null 2>&1; then
    # Use GNU parallel for efficient job control
    printf "%s\n" "${JOBS[@]}" | parallel -j "$NUM_CPUS" --joblog run_joblog.log --lb --colsep '\t' 'run_job {#} {= $_ = $_ =}'
    EXIT_CODE=$?
else
    # Manual bash job control
    echo "GNU parallel not found. Using bash job control."
    PIDS=()
    JOB_IDX=0
    NEXT_JOB=0
    FINISHED=0
    declare -A JOB_STATUS

    # Start up to NUM_CPUS jobs
    while [[ $NEXT_JOB -lt $TOTAL_JOBS || ${#PIDS[@]} -gt 0 ]]; do
        # Launch new jobs if we have CPUs available
        while [[ $NEXT_JOB -lt $TOTAL_JOBS && ${#PIDS[@]} -lt $NUM_CPUS ]]; do
            run_job "$NEXT_JOB" "${JOBS[$NEXT_JOB]}" &
            PIDS+=("$!")
            JOB_STATUS["$!"]=$NEXT_JOB
            ((NEXT_JOB++))
        done
        # Wait for any job to finish
        if [[ ${#PIDS[@]} -gt 0 ]]; then
            wait -n
            # Remove finished PIDs
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                else
                    ((FINISHED++))
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
        fi
    done
    EXIT_CODE=0
fi

echo "All jobs completed."
exit $EXIT_CODE 