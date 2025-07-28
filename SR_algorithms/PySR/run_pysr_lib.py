#!/usr/bin/env python3
"""
PySR Symbolic Regression with Minimal Library Configuration
This script runs PySR with problem-specific minimal function libraries.
"""

import sys
import os
import csv
import pandas as pd
import numpy as np
import glob
import time
import argparse

try:
    from pysr import PySRRegressor
except ImportError:
    print("PySR is not installed in this environment. Please install it with 'pip install pysr'.")
    sys.exit(1)

def load_minimal_config(problem_type):
    """
    Load minimal configuration for the specified problem type.
    
    Args:
        problem_type (str): 'one_max', 'leading_ones', or 'psa'
    
    Returns:
        tuple: (binary_operators, unary_operators)
    """
    config_dir = os.path.join(os.path.dirname(__file__), 'configs')
    
    if problem_type == 'one_max':
        config_file = os.path.join(config_dir, 'minimal_onemax.py')
    elif problem_type == 'leading_ones':
        config_file = os.path.join(config_dir, 'minimal_leadingones.py')
    elif problem_type == 'psa':
        config_file = os.path.join(config_dir, 'minimal_psacmaes.py')
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Load configuration by executing the config file
    config_globals = {}
    with open(config_file, 'r') as f:
        exec(f.read(), config_globals)
    
    return config_globals['BINARY_OPERATORS'], config_globals['UNARY_OPERATORS']

def extract_best_equation_from_hof(hof_path):
    try:
        hof_df = pd.read_csv(hof_path)
        if 'Loss' in hof_df.columns and 'Equation' in hof_df.columns:
            best_row = hof_df.loc[hof_df['Loss'].idxmin()]
            return str(best_row['Equation']), float(best_row['Loss'])
        else:
            print(f"hall_of_fame.csv at {hof_path} missing required columns.")
            return "ERROR: hall_of_fame.csv missing columns", float('inf')
    except Exception as e:
        print(f"Error reading hall_of_fame.csv: {e}")
        return "ERROR: Could not read hall_of_fame.csv", float('inf')

def get_best_loss_from_hof(outputs_dir):
    """Get the best loss from the most recent hall_of_fame.csv"""
    try:
        # Find all hall_of_fame.csv files in subdirectories
        hof_files = glob.glob(os.path.join(outputs_dir, "*/hall_of_fame.csv"))
        if hof_files:
            # Use the most recently modified one
            hof_path = max(hof_files, key=os.path.getmtime)
            _, best_loss = extract_best_equation_from_hof(hof_path)
            return best_loss
        else:
            return float('inf')
    except Exception as e:
        print(f"Error getting best loss: {e}")
        return float('inf')

def run_pysr_with_minimal_library(X, y, problem_type, noise_threshold, max_iterations=100, min_iterations=10):
    """
    Run PySR with minimal library configuration.
    
    Args:
        X: Input features
        y: Target values
        problem_type: Type of problem ('one_max', 'leading_ones', 'psa')
        noise_threshold: Convergence threshold (noise level)
        max_iterations: Maximum number of iterations
        min_iterations: Minimum number of iterations before stopping
    
    Returns:
        tuple: (equation, final_loss, iterations_used)
    """
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    
    # Load minimal configuration
    binary_operators, unary_operators = load_minimal_config(problem_type)
    
    # Calculate optimal parameters based on dataset size and tuning guide
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Set parsimony based on expected minimum loss (noise_threshold / 5-10)
    parsimony = max(noise_threshold / 8.0, 1e-8)
    
    # Optimize population size for single CPU
    population_size = 50  # Conservative for single CPU
    
    # Set ncycles_per_iteration for single CPU (lower than cluster settings)
    ncycles_per_iteration = 1000  # Good balance for single CPU
    
    # Adaptive settings based on dataset size
    if n_samples > 1000:
        # For large datasets, use batching and adjust parameters
        use_batching = True
        population_size = min(population_size, 30)  # Smaller population for large datasets
    else:
        use_batching = False
    
    print(f"Running PySR with minimal library for {problem_type}")
    print(f"Binary operators: {binary_operators}")
    print(f"Unary operators: {unary_operators}")
    print(f"Convergence threshold: {noise_threshold}")
    print(f"Dataset size: {n_samples} samples, {n_features} features")
    print(f"Parsimony: {parsimony:.2e}, Population: {population_size}, Cycles: {ncycles_per_iteration}")
    print(f"Batching: {use_batching}, Turbo: True")
    
    # Create a unique output directory for this run to avoid conflicts in parallel execution
    import uuid
    unique_output_dir = f"outputs_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    
    # Create fresh PySR model with minimal library for each run
    # This ensures fairness in experiments by avoiding any cached state
    model = PySRRegressor(
        niterations=1,  # We'll control iterations manually
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        progress=False,  # Disable progress bar for cleaner output
        model_selection="best",
        maxsize=25,  # Increased from 20 for better exploration
        procs=1,  # Single CPU for fairness
        output_directory=unique_output_dir,
        # Optimized parameters from tuning guide
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        parsimony=parsimony,
        weight_optimize=0.001,  # Important for optimization frequency
        turbo=True,  # 20%+ speedup with advanced loop vectorization
        # Dataset optimization
        batching=use_batching,  # Use batching for large datasets
        # Complexity settings - adjust based on available operators
        complexity_of_operators={op: 2.0 for op in unary_operators} if unary_operators else None,
        # Additional robustness settings
        adaptive_parsimony_scaling=100,  # Helps with exploration
        warmup_maxsize_by=0.5,  # Start with smaller expressions
        # Loss function optimization
        loss="L2DistLoss()",  # Standard L2 loss for regression
        # Precision settings
        precision=32  # Use 32-bit precision for speed
    )
    
    print(f"Max iterations: {max_iterations}, Min iterations: {min_iterations}")
    
    # Start training
    start_time = time.time()
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Run one iteration
        try:
            # Fit for one iteration
            model.fit(X, y)
            
            # Check convergence
            best_loss = get_best_loss_from_hof(unique_output_dir)
            
            print(f"  Current best loss: {best_loss:.6f}")
            
            # Check if converged
            if iteration >= min_iterations - 1 and best_loss < noise_threshold:
                print(f"  Converged at iteration {iteration + 1} with loss {best_loss:.6f}")
                break
                
        except Exception as e:
            print(f"  Error in iteration {iteration + 1}: {e}")
            break
    
    # Get final results from the unique output directory
    hof_files = glob.glob(os.path.join(unique_output_dir, "*/hall_of_fame.csv"))
    if hof_files:
        hof_path = max(hof_files, key=os.path.getmtime)
        equation, final_loss = extract_best_equation_from_hof(hof_path)
    else:
        equation = "ERROR: No equation found"
        final_loss = float('inf')
    
    elapsed_time = time.time() - start_time
    print(f"PySR completed in {elapsed_time:.2f} seconds")
    print(f"Final loss: {final_loss:.6f}")
    
    # Clean up the unique output directory after extracting results
    # This ensures fairness in experiments and prevents accumulation of temporary files
    try:
        import shutil
        if os.path.exists(unique_output_dir):
            shutil.rmtree(unique_output_dir)
            print(f"Cleaned up output directory: {unique_output_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up output directory {unique_output_dir}: {e}")
    
    return equation, final_loss, iteration + 1

def main():
    parser = argparse.ArgumentParser(description="Run PySR symbolic regression with minimal library")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("problem_type", help="Problem type: one_max, leading_ones, or psa")
    parser.add_argument("--noise", type=float, default=1e-12, help="Noise threshold for convergence (default: 1e-12)")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations (default: 100)")
    parser.add_argument("--min-iterations", type=int, default=10, help="Minimum iterations before stopping (default: 10)")
    args = parser.parse_args()
    
    csv_file = args.csv_file
    problem_type = args.problem_type
    noise_threshold = args.noise
    max_iterations = args.max_iterations
    min_iterations = args.min_iterations
    
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        sys.exit(1)
    
    # Validate problem type
    valid_problems = ['one_max', 'leading_ones', 'psa']
    if problem_type not in valid_problems:
        print(f"Error: Problem type must be one of {valid_problems}")
        sys.exit(1)
    
    # Load the dataset
    try:
        data = pd.read_csv(csv_file, header=None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Assume the last column is the target, the rest are features
    if data.shape[1] < 2:
        print("Error: CSV file must have at least one feature column and one target column.")
        sys.exit(1)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    print(f"Running PySR symbolic regression on: {csv_file}")
    print(f"Problem type: {problem_type}")
    print(f"Dataset shape: {X.shape}")
    
    # Run PySR with minimal library
    equation, final_loss, iterations_used = run_pysr_with_minimal_library(
        X, y, problem_type, noise_threshold, max_iterations, min_iterations
    )
    
    # Update or create results_lib.csv in the same directory as the input CSV
    csv_dir = os.path.dirname(csv_file)
    results_file = os.path.join(csv_dir, "results_lib.csv")
    
    try:
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            if 'pysr' not in results_df.columns:
                results_df['pysr'] = ''
            # Always update the first row
            if len(results_df) > 0:
                results_df.at[0, 'pysr'] = equation
            else:
                # If the file is empty but has columns, create a single row
                empty_row = {col: '' for col in results_df.columns}
                empty_row['pysr'] = equation
                results_df = pd.DataFrame([empty_row])
        else:
            # If the file doesn't exist, create a single-row DataFrame with only pysr
            results_df = pd.DataFrame({'pysr': [equation]})
        
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        print(f"Final equation: {equation}")
        print(f"Convergence summary: {iterations_used} iterations, final loss: {final_loss:.6f}")
        
    except Exception as e:
        print(f"Error updating results file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 