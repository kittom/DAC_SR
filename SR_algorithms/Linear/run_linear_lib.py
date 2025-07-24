#!/usr/bin/env python3
"""
Linear Regression with Minimal Library Configuration
Note: Linear regression doesn't actually need minimal library since it's just linear,
but we keep the interface consistent with other algorithms.
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_linear_with_minimal_library(csv_file, problem_type, noise_threshold):
    """Run linear regression (minimal library version)."""
    
    print(f"Running linear regression for {problem_type}")
    print(f"Note: Linear regression doesn't use minimal library - it's always linear")
    print(f"Noise threshold: {noise_threshold} (not used for linear regression)")
    
    # Load the CSV data
    try:
        data = pd.read_csv(csv_file, header=None)
        print(f"Loaded {len(data)} data points")
        
        # Use all columns except the last as features (X), last column as target (y)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return "<error_loading_data>", float('inf'), 0
    
    # Create and fit linear regression model
    try:
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Create equation string
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Build equation string
        equation_parts = []
        for i, coef in enumerate(coefficients):
            if coef != 0:
                if coef == 1:
                    equation_parts.append(f"x{i+1}")
                elif coef == -1:
                    equation_parts.append(f"-x{i+1}")
                else:
                    equation_parts.append(f"{coef:.6f}*x{i+1}")
        
        if intercept != 0:
            equation_parts.append(f"{intercept:.6f}")
        
        if not equation_parts:
            equation = "0"
        else:
            equation = " + ".join(equation_parts)
        
        print(f"Linear equation: {equation}")
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.6f}")
        
        return equation, mse, 1
        
    except Exception as e:
        print(f"Error running linear regression: {e}")
        return "<error_running_linear>", float('inf'), 0

def main():
    parser = argparse.ArgumentParser(description="Run linear regression (minimal library version)")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("problem_type", help="Problem type: one_max, leading_ones, or psa")
    parser.add_argument("--noise", type=float, default=1e-12, help="Noise threshold (not used for linear regression)")
    args = parser.parse_args()
    
    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found!")
        sys.exit(1)
    
    # Run linear regression
    equation, final_loss, iterations_used = run_linear_with_minimal_library(
        args.csv_file, args.problem_type, args.noise
    )
    
    # Update results_lib.csv
    csv_dir = os.path.dirname(args.csv_file)
    results_file = os.path.join(csv_dir, "results_lib.csv")
    
    try:
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            if 'linear' not in results_df.columns:
                results_df['linear'] = ''
            if len(results_df) > 0:
                results_df.at[0, 'linear'] = equation
        else:
            results_df = pd.DataFrame({'linear': [equation]})
        
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error updating results file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 