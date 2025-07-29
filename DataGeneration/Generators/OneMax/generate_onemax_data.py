#!/usr/bin/env python3
"""
Consolidated OneMax data generation script.
Combines standard, hidden variables, and noise functionality.
"""

import pandas as pd
import numpy as np
import os
import sys
import math
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_one_max_data(instance_sizes=None, data_type='continuous', output_dir=None, 
                         evaluation_type=None, hidden_variables=False, noise_level=0.0, 
                         noise_type='gaussian', dropout_rate=0.0):
    """
    Generate OneMax ground truth data with various options.
    
    Args:
        instance_sizes: List of instance sizes to generate data for
        data_type: 'continuous' or 'discrete'
        output_dir: Output directory path
        evaluation_type: 'control', 'library', 'rounding', or None for all
        hidden_variables: If True, hide instance size from dataset
        noise_level: Noise level to add (0.0 for no noise)
        noise_type: 'gaussian' or 'uniform' noise
        dropout_rate: Fraction of rows to randomly remove (0.0 for no dropout)
    """
    
    if instance_sizes is None:
        instance_sizes = [10, 20, 30, 40, 50, 100]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    instance_size_data = []
    current_state_data = []
    bitflip_data = []
    
    for instance_size in instance_sizes:
        for current_state in range(instance_size + 1):
            # OneMax uses sqrt(instance_size/(instance_size-current_state))
            if current_state == instance_size:
                # Replace infinite values with a large finite value for compatibility
                theoretical_bitflip = 0.0
            else:
                theoretical_bitflip = math.sqrt(instance_size / (instance_size - current_state))
            
            # Add noise if specified
            if noise_level > 0.0:
                if noise_type == 'gaussian':
                    noise = np.random.normal(0, noise_level)
                elif noise_type == 'uniform':
                    noise = np.random.uniform(-noise_level, noise_level)
                else:
                    raise ValueError("noise_type must be 'gaussian' or 'uniform'")
                bitflip = theoretical_bitflip + noise
                bitflip = max(0.001, bitflip)  # Ensure positive values
            else:
                bitflip = theoretical_bitflip
            
            # Round for discrete data
            if data_type == 'discrete':
                bitflip = round(bitflip)
            
            # Store data
            if not hidden_variables:
                instance_size_data.append(instance_size)
            current_state_data.append(current_state)
            bitflip_data.append(bitflip)
    
    # Create DataFrame
    if hidden_variables:
        df = pd.DataFrame({
            'CurrentState': current_state_data,
            'Bitflip': bitflip_data
        })
    else:
        df = pd.DataFrame({
            'InstanceSize': instance_size_data,
            'CurrentState': current_state_data,
            'Bitflip': bitflip_data
        })
    
    # Apply dropout if specified
    if dropout_rate > 0.0:
        original_rows = len(df)
        rows_to_remove = int(original_rows * dropout_rate)
        if rows_to_remove > 0:
            rows_to_drop = np.random.choice(original_rows, rows_to_remove, replace=False)
            df = df.drop(df.index[rows_to_drop]).reset_index(drop=True)
            print(f"Applied dropout: removed {rows_to_remove}/{original_rows} rows ({dropout_rate*100:.1f}%)")
    
    # Save to output directory
    if output_dir is None:
        if hidden_variables:
            out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth_Hidden/OneMax/{data_type}')
        elif noise_level > 0.0:
            out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth_Noise/OneMax/{data_type}')
        else:
            out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/OneMax/{data_type}')
    else:
        out_dir = output_dir
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'GTOneMax.csv')
    df.to_csv(out_path, index=False, header=False)
    print(f"Output saved to: {out_path}")
    
    # Create ground truth equation
    if hidden_variables:
        # For hidden variables, use a fixed instance size in the formula
        ground_truth_equation = "sqrt(500/(500-x1))" if data_type == 'continuous' else "round(sqrt(500/(500-x1)))"
    else:
        ground_truth_equation = "sqrt(x1/(x1-x2))" if data_type == 'continuous' else "round(sqrt(x1/(x1-x2)))"
    
    # Create results files based on evaluation type
    if evaluation_type == 'control' or evaluation_type is None:
        control_results_path = os.path.join(out_dir, 'results.csv')
        control_results_data = {'ground_truth': [ground_truth_equation]}
        control_results_df = pd.DataFrame(control_results_data)
        control_results_df.to_csv(control_results_path, index=False)
        print(f"Control library results file created: {control_results_path}")
    
    if evaluation_type == 'library' or evaluation_type is None:
        tailored_results_path = os.path.join(out_dir, 'results_lib.csv')
        tailored_results_data = {'ground_truth': [ground_truth_equation]}
        tailored_results_df = pd.DataFrame(tailored_results_data)
        tailored_results_df.to_csv(tailored_results_path, index=False)
        print(f"Tailored library results file created: {tailored_results_path}")
    
    if evaluation_type == 'rounding' or evaluation_type is None:
        rounding_results_path = os.path.join(out_dir, 'results_rounding.csv')
        rounding_results_data = {'ground_truth': [ground_truth_equation]}
        rounding_results_df = pd.DataFrame(rounding_results_data)
        rounding_results_df.to_csv(rounding_results_path, index=False)
        print(f"Rounding results file created: {rounding_results_path}")
    
    print(f"Ground truth equation: {ground_truth_equation}")
    print(f"Generated {len(df)} rows of data")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate OneMax ground truth data")
    parser.add_argument('--instance-sizes', type=int, nargs='*', default=[10, 20, 30, 40, 50, 100],
                       help='Instance sizes to generate data for')
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous',
                       help='Type of data to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for the generated data')
    parser.add_argument('--evaluation-type', choices=['control', 'library', 'rounding'], default=None,
                       help='Type of evaluation to prepare results for')
    parser.add_argument('--hidden-variables', action='store_true',
                       help='Hide instance size from dataset (use fixed size in formula)')
    parser.add_argument('--noise-level', type=float, default=0.0,
                       help='Noise level (standard deviation for gaussian, range for uniform)')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian',
                       help='Type of noise to add')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                       help='Fraction of rows to randomly remove (0.0-1.0)')
    
    args = parser.parse_args()
    
    print(f"Generating OneMax data with parameters:")
    print(f"  Instance sizes: {args.instance_sizes}")
    print(f"  Data type: {args.data_type}")
    print(f"  Hidden variables: {args.hidden_variables}")
    print(f"  Noise level: {args.noise_level}")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Evaluation type: {args.evaluation_type}")
    
    df = generate_one_max_data(
        instance_sizes=args.instance_sizes,
        data_type=args.data_type,
        output_dir=args.output_dir,
        evaluation_type=args.evaluation_type,
        hidden_variables=args.hidden_variables,
        noise_level=args.noise_level,
        noise_type=args.noise_type,
        dropout_rate=args.dropout_rate
    )
    
    print("OneMax data generation completed successfully!")

if __name__ == "__main__":
    main() 