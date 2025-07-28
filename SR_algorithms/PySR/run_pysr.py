#!/usr/bin/env python3
"""
PySR Symbolic Regression with Convergence-Based Stopping
This script runs PySR with early stopping based on loss convergence.
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

def run_pysr_with_convergence(X, y, noise_threshold, max_iterations=100, min_iterations=10):
    """
    Run PySR with convergence-based stopping.
    
    Args:
        X: Input features
        y: Target values
        noise_threshold: Convergence threshold (noise level)
        max_iterations: Maximum number of iterations
        min_iterations: Minimum number of iterations before stopping
    
    Returns:
        tuple: (equation, final_loss, iterations_used)
    """
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    
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
    
    # Create PySR model with optimized settings
    model = PySRRegressor(
        niterations=1,  # We'll control iterations manually
        binary_operators=["+", "-", "*", "/", "^"],  # Keep power operator
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],  # Removed redundant tan, tanh
        progress=False,  # Disable progress bar for cleaner output
        model_selection="best",
        maxsize=25,  # Increased from 20 for better exploration
        procs=1,  # Single CPU for fairness
        output_directory="outputs",
        # Optimized parameters from tuning guide
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        parsimony=parsimony,
        weight_optimize=0.001,  # Important for optimization frequency
        turbo=True,  # 20%+ speedup with advanced loop vectorization
        # Constraints to reduce redundant expressions
        constraints={"pow": (9, 1)},  # Power laws only with variable/constant exponents
        nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},  # No nested trig
        # Dataset optimization
        batching=use_batching,  # Use batching for large datasets
        # Complexity settings
        complexity_of_operators={"pow": 3, "sin": 2, "cos": 2, "exp": 2, "log": 2, "sqrt": 1.5, "abs": 1},
        # Additional robustness settings
        adaptive_parsimony_scaling=100,  # Helps with exploration
        warmup_maxsize_by=0.5,  # Start with smaller expressions
        # Loss function optimization
        loss="L2DistLoss()",  # Standard L2 loss for regression
        # Precision settings
        precision=32  # Use 32-bit precision for speed
    )
    
    print(f"Running PySR with convergence threshold: {noise_threshold}")
    print(f"Max iterations: {max_iterations}, Min iterations: {min_iterations}")
    print(f"Dataset size: {n_samples} samples, {n_features} features")
    print(f"Parsimony: {parsimony:.2e}, Population: {population_size}, Cycles: {ncycles_per_iteration}")
    print(f"Batching: {use_batching}, Turbo: True")
    
    # Start training
    start_time = time.time()
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Run one iteration
        try:
            # Fit for one iteration
            model.fit(X, y)
            
            # Check convergence
            best_loss = get_best_loss_from_hof(outputs_dir)
            
            print(f"  Current best loss: {best_loss:.6f}")
            
            # Check if converged
            if iteration >= min_iterations - 1 and best_loss < noise_threshold:
                print(f"  Converged at iteration {iteration + 1} with loss {best_loss:.6f}")
                break
                
        except Exception as e:
            print(f"  Error in iteration {iteration + 1}: {e}")
            break
    
    # Get final results
    hof_files = glob.glob(os.path.join(outputs_dir, "*/hall_of_fame.csv"))
    if hof_files:
        hof_path = max(hof_files, key=os.path.getmtime)
        equation, final_loss = extract_best_equation_from_hof(hof_path)
    else:
        equation = "ERROR: No equation found"
        final_loss = float('inf')
    
    elapsed_time = time.time() - start_time
    print(f"PySR completed in {elapsed_time:.2f} seconds")
    print(f"Final loss: {final_loss:.6f}")
    
    return equation, final_loss, iteration + 1

def main():
    parser = argparse.ArgumentParser(description="Run PySR symbolic regression with convergence-based stopping")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--noise", type=float, default=1e-12, help="Noise threshold for convergence (default: 1e-12)")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations (default: 100)")
    parser.add_argument("--min-iterations", type=int, default=10, help="Minimum iterations before stopping (default: 10)")
    args = parser.parse_args()
    
    csv_file = args.csv_file
    noise_threshold = args.noise
    max_iterations = args.max_iterations
    min_iterations = args.min_iterations
    
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
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
    print(f"Dataset shape: {X.shape}")
    
    # Run PySR with convergence-based stopping
    equation, final_loss, iterations_used = run_pysr_with_convergence(
        X, y, noise_threshold, max_iterations, min_iterations
    )
    
    # Update or create results.csv in the same directory as the input CSV
    csv_dir = os.path.dirname(csv_file)
    results_file = os.path.join(csv_dir, "results.csv")
    
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