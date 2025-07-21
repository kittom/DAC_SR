#!/usr/bin/env python3
"""
Test script to demonstrate how to use the generated CMA-ES ground truth data.
This script shows how to load the data and use it for training or evaluation.
"""

import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def load_cmaes_data(data_type='continuous'):
    """Load the generated CMA-ES data."""
    data_path = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/CMAES/{data_type}/GTCMAES.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    # Load data with column names
    columns = ['FunctionID', 'Dimension', 'Repetition', 'Algorithm', 'Lambda', 'PtNormLog', 'ScaleFactor', 'Precision', 'UsedBudget']
    df = pd.read_csv(data_path, header=None, names=columns)
    
    return df

def load_optimal_policy_data():
    """Load the optimal policy data."""
    data_path = os.path.join(PROJECT_ROOT, 'DataSets/Ground_Truth/CMAES/continuous/GTCMAES_OptimalPolicy.csv')
    
    if not os.path.exists(data_path):
        print(f"Optimal policy file not found: {data_path}")
        return None
    
    # Load data with column names
    columns = ['FunctionID', 'Dimension', 'Lambda', 'PtNormLog', 'ScaleFactor', 'OptimalLambda']
    df = pd.read_csv(data_path, header=None, names=columns)
    
    return df

def demonstrate_data_usage():
    """Demonstrate how to use the CMA-ES data."""
    print("=== CMA-ES Data Usage Demonstration ===\n")
    
    # Load the data
    print("1. Loading CMA-ES ground truth data...")
    df = load_cmaes_data('continuous')
    if df is None:
        return
    
    print(f"   Loaded {len(df)} data points")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Show data structure
    print("2. Data structure overview:")
    print(f"   Functions: {sorted(df['FunctionID'].unique())}")
    print(f"   Dimensions: {sorted(df['Dimension'].unique())}")
    print(f"   Algorithms: {sorted(df['Algorithm'].unique())}")
    print(f"   Repetitions: {sorted(df['Repetition'].unique())}")
    print()
    
    # Show state space ranges
    print("3. State space ranges:")
    print(f"   Lambda (population size): [{df['Lambda'].min():.1f}, {df['Lambda'].max():.1f}]")
    print(f"   PtNorm (log scale): [{df['PtNormLog'].min():.3f}, {df['PtNormLog'].max():.3f}]")
    print(f"   Scale Factor: [{df['ScaleFactor'].min():.3f}, {df['ScaleFactor'].max():.3f}]")
    print()
    
    # Show performance comparison
    print("4. Algorithm performance comparison:")
    final_precision = df.groupby(['Algorithm', 'FunctionID', 'Dimension', 'Repetition'])['Precision'].last()
    for algorithm in sorted(df['Algorithm'].unique()):
        algo_precision = final_precision.xs(algorithm, level=0)
        print(f"   {algorithm.upper()}:")
        print(f"     Mean final precision: {algo_precision.mean():.2e}")
        print(f"     Std final precision: {algo_precision.std():.2e}")
        print(f"     Mean population size: {df[df['Algorithm'] == algorithm]['Lambda'].mean():.1f}")
    print()
    
    # Load optimal policy data
    print("5. Loading optimal policy data...")
    optimal_df = load_optimal_policy_data()
    if optimal_df is not None:
        print(f"   Loaded {len(optimal_df)} optimal policy data points")
        print(f"   Optimal lambda range: [{optimal_df['OptimalLambda'].min():.1f}, {optimal_df['OptimalLambda'].max():.1f}]")
        print(f"   Mean optimal lambda: {optimal_df['OptimalLambda'].mean():.1f}")
    print()
    
    # Demonstrate how to extract training data
    print("6. Extracting training data for a specific function and dimension:")
    fid = 1
    dim = 10
    training_data = df[(df['FunctionID'] == fid) & (df['Dimension'] == dim)]
    
    print(f"   Function {fid}, Dimension {dim}: {len(training_data)} data points")
    
    # Extract state-action pairs
    states = training_data[['Lambda', 'PtNormLog', 'ScaleFactor']].values
    actions = training_data['Lambda'].values  # Current lambda as action (for demonstration)
    
    print(f"   State shape: {states.shape}")
    print(f"   Action shape: {actions.shape}")
    print(f"   State sample: {states[0]}")
    print(f"   Action sample: {actions[0]}")
    print()
    
    # Demonstrate how to use optimal policy data
    if optimal_df is not None:
        print("7. Using optimal policy data:")
        optimal_for_fid_dim = optimal_df[(optimal_df['FunctionID'] == fid) & (optimal_df['Dimension'] == dim)]
        
        print(f"   Optimal policy data points for FID {fid}, dim {dim}: {len(optimal_for_fid_dim)}")
        
        # Find optimal action for a specific state
        target_state = np.array([100.0, 2.0, 5.0])  # Example state
        print(f"   Target state: {target_state}")
        
        # Find closest state in optimal policy data
        state_diffs = np.linalg.norm(optimal_for_fid_dim[['Lambda', 'PtNormLog', 'ScaleFactor']].values - target_state, axis=1)
        closest_idx = np.argmin(state_diffs)
        closest_state = optimal_for_fid_dim.iloc[closest_idx]
        
        print(f"   Closest state in optimal policy: {closest_state[['Lambda', 'PtNormLog', 'ScaleFactor']].values}")
        print(f"   Optimal action: {closest_state['OptimalLambda']:.1f}")
    print()
    
    print("=== Demonstration Complete ===")

def demonstrate_training_setup():
    """Demonstrate how to set up training data for a learning algorithm."""
    print("=== Training Setup Demonstration ===\n")
    
    # Load data
    df = load_cmaes_data('continuous')
    optimal_df = load_optimal_policy_data()
    
    if df is None or optimal_df is None:
        print("Could not load data for training setup demonstration.")
        return
    
    # Create training dataset
    print("1. Creating training dataset...")
    
    # For this demonstration, we'll use the optimal policy data as training data
    # In practice, you might want to use the raw data and learn from experience
    
    # Prepare state-action pairs
    X = optimal_df[['Lambda', 'PtNormLog', 'ScaleFactor']].values  # States
    y = optimal_df['OptimalLambda'].values  # Optimal actions
    
    print(f"   Training data shape: X={X.shape}, y={y.shape}")
    print(f"   State range: [{X.min(axis=0)}, {X.max(axis=0)}]")
    print(f"   Action range: [{y.min():.1f}, {y.max():.1f}]")
    print()
    
    # Split by function and dimension for cross-validation
    print("2. Creating cross-validation splits...")
    
    unique_configs = optimal_df[['FunctionID', 'Dimension']].drop_duplicates()
    print(f"   Number of function-dimension configurations: {len(unique_configs)}")
    
    for _, config in unique_configs.iterrows():
        fid, dim = config['FunctionID'], config['Dimension']
        config_data = optimal_df[(optimal_df['FunctionID'] == fid) & (optimal_df['Dimension'] == dim)]
        print(f"   FID {fid}, dim {dim}: {len(config_data)} samples")
    
    print()
    
    # Demonstrate how to create a simple policy function
    print("3. Creating a simple policy function...")
    
    def simple_policy(state, optimal_data):
        """
        Simple policy that finds the closest state in optimal data and returns optimal action.
        
        Parameters:
        -----------
        state : np.array
            Current state [lambda, pt_norm_log, scale_factor]
        optimal_data : pd.DataFrame
            Optimal policy data
            
        Returns:
        --------
        float : Optimal population size
        """
        # Find closest state
        state_diffs = np.linalg.norm(optimal_data[['Lambda', 'PtNormLog', 'ScaleFactor']].values - state, axis=1)
        closest_idx = np.argmin(state_diffs)
        return optimal_data.iloc[closest_idx]['OptimalLambda']
    
    # Test the policy
    test_state = np.array([150.0, 1.5, 3.0])
    optimal_action = simple_policy(test_state, optimal_df)
    
    print(f"   Test state: {test_state}")
    print(f"   Predicted optimal action: {optimal_action:.1f}")
    print()
    
    print("=== Training Setup Complete ===")

if __name__ == "__main__":
    print("CMA-ES Data Usage Test Script")
    print("=" * 50)
    
    demonstrate_data_usage()
    print("\n" + "=" * 50 + "\n")
    demonstrate_training_setup() 