#!/usr/bin/env python3
"""
Linear Regression Algorithm for Symbolic Regression Comparison
This script performs linear regression on CSV data and outputs the learned equation.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import argparse

def run_linear_regression(csv_file):
    """
    Run linear regression on the provided CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing the data
        
    Returns:
        str: The learned linear equation
    """
    try:
        # Read the CSV file (assuming no headers)
        data = pd.read_csv(csv_file, header=None)
        
        # Separate features (all columns except the last) and target (last column)
        X = data.iloc[:, :-1].values  # All columns except the last
        y = data.iloc[:, -1].values   # Last column is the target
        
        # Standardize features for better numerical stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Get coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Create the equation string
        equation_parts = []
        
        # Add intercept if it's not zero
        if abs(intercept) > 1e-10:
            equation_parts.append(f"{intercept:.6f}")
        
        # Add coefficient terms
        for i, coef in enumerate(coefficients):
            if abs(coef) > 1e-10:  # Only include non-zero coefficients
                var_name = f"x{i+1}"
                if coef > 0 and equation_parts:  # Add + sign for positive coefficients (except first term)
                    equation_parts.append(f"+ {coef:.6f} * {var_name}")
                else:
                    equation_parts.append(f"{coef:.6f} * {var_name}")
        
        # Join all parts
        equation = " + ".join(equation_parts)
        
        # If equation is empty, return 0
        if not equation:
            equation = "0"
            
        return equation
        
    except Exception as e:
        print(f"Error in linear regression: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description="Run linear regression on CSV data")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()
    
    csv_file = args.csv_file
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        sys.exit(1)
    
    print(f"Running linear regression on: {csv_file}")
    
    # Run linear regression
    equation = run_linear_regression(csv_file)
    
    # Get the directory of the CSV file
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    results_file = os.path.join(csv_dir, "results.csv")
    
    # Read existing results if they exist
    existing_results = {}
    if os.path.exists(results_file):
        try:
            results_df = pd.read_csv(results_file)
            if 'ground_truth' in results_df.columns:
                existing_results['ground_truth'] = results_df['ground_truth'].iloc[0]
        except:
            pass
    
    # Create new results DataFrame
    results_data = {}
    
    # Add ground truth if it exists
    if 'ground_truth' in existing_results:
        results_data['ground_truth'] = [existing_results['ground_truth']]
    
    # Add linear regression result
    results_data['linear'] = [equation]
    
    # Save results
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_file, index=False)
    
    print(f"Linear regression equation: {equation}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main() 