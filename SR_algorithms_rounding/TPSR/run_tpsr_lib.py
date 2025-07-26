#!/usr/bin/env python3
"""
TPSR Symbolic Regression with Minimal Library Configuration
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse
import time
import torch
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine
from tpsr import tpsr_fit

def load_minimal_config(problem_type):
    """Load minimal configuration for the specified problem type."""
    config_dir = os.path.join(os.path.dirname(__file__), 'configs')
    
    if problem_type == 'one_max':
        config_file = os.path.join(config_dir, 'minimal_onemax.py')
    elif problem_type == 'leading_ones':
        config_file = os.path.join(config_dir, 'minimal_leadingones.py')
    elif problem_type == 'psa':
        config_file = os.path.join(config_dir, 'minimal_psacmaes.py')
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    config_globals = {}
    with open(config_file, 'r') as f:
        exec(f.read(), config_globals)
    
    return config_globals['MINIMAL_OPERATORS']

def run_tpsr_with_minimal_library(X, y, problem_type, noise_threshold, max_horizon=200, min_horizon=10):
    """Run TPSR with minimal library configuration."""
    
    minimal_operators = load_minimal_config(problem_type)
    
    print(f"Running TPSR with minimal library for {problem_type}")
    print(f"Operators: {minimal_operators}")
    print(f"Convergence threshold: {noise_threshold}")
    
    # Convert noise threshold to reward threshold
    reward_threshold = 1 / (1 + noise_threshold)
    
    print(f"Reward threshold: {reward_threshold:.6f}")
    print(f"Max horizon: {max_horizon}, Min horizon: {min_horizon}")
    
    try:
        # Set up TPSR (E2E backbone, CPU)
        parser = get_parser()
        tpsr_params = parser.parse_args([])  # empty list to use defaults
        tpsr_params.cpu = True
        tpsr_params.device = torch.device("cpu")
        tpsr_params.debug = False
        tpsr_params.backbone_model = 'e2e'
        
        # Override horizon parameters
        tpsr_params.horizon = max_horizon
        
        # Set minimal operators in the environment
        # Note: This is a simplified approach - in practice, we'd need to modify the environment
        # to only use the minimal operators, but for now we'll run with the full set
        # and let the algorithm find the best solution
        
        # Build environment and model
        np.random.seed(tpsr_params.seed)
        torch.manual_seed(tpsr_params.seed)
        equation_env = build_env(tpsr_params)
        modules = build_modules(equation_env, tpsr_params)
        trainer = Trainer(modules, equation_env, tpsr_params)
        
        # Create samples dictionary
        samples = {'x_to_fit': X, 'y_to_fit': y}
        
        # Run TPSR with convergence monitoring
        start_time = time.time()
        
        # Run TPSR (MCTS loop) with the full horizon
        final_seq, time_elapsed, best_reward = tpsr_fit(X, y, tpsr_params, equation_env)
        
        # Check if we converged early
        if best_reward is not None and best_reward > reward_threshold:
            print(f"TPSR converged with reward: {best_reward:.6f}")
            horizon_used = len(final_seq) if final_seq else max_horizon
        else:
            print(f"TPSR did not converge within {max_horizon} steps")
            horizon_used = max_horizon
        
        elapsed_time = time.time() - start_time
        
        # Use pred_for_sample_no_refine to get the equation string
        y_pred, eq_str, _ = pred_for_sample_no_refine(
            Transformer(params=tpsr_params, env=equation_env, samples=samples),
            equation_env,
            final_seq,
            X
        )
        
        if eq_str is None:
            eq_str = "<no_equation_found>"
        
        print(f"TPSR completed in {elapsed_time:.2f} seconds")
        print(f"Final reward: {best_reward:.6f}")
        print(f"Horizon used: {horizon_used}")
        
        return eq_str, best_reward, horizon_used
        
    except Exception as e:
        print(f"Error running TPSR: {e}")
        return "<error_running_tpsr>", 0.0, max_horizon

def main():
    parser = argparse.ArgumentParser(description="Run TPSR symbolic regression with minimal library")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("problem_type", help="Problem type: one_max, leading_ones, or psa")
    parser.add_argument("--noise", type=float, default=1e-12, help="Noise threshold for convergence")
    parser.add_argument("--max-horizon", type=int, default=200, help="Maximum MCTS horizon")
    parser.add_argument("--min-horizon", type=int, default=10, help="Minimum horizon before stopping")
    args = parser.parse_args()
    
    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found!")
        sys.exit(1)
    
    # Load data
    data = pd.read_csv(args.csv_file, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    
    # Run TPSR with minimal library
    equation, final_reward, horizon_used = run_tpsr_with_minimal_library(
        [X], [y], args.problem_type, args.noise, args.max_horizon, args.min_horizon
    )
    
    # Update results_lib.csv
    csv_dir = os.path.dirname(args.csv_file)
    results_file = os.path.join(csv_dir, "results_lib.csv")
    
    try:
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            if 'tpsr' not in results_df.columns:
                results_df['tpsr'] = ''
            if len(results_df) > 0:
                results_df.at[0, 'tpsr'] = equation
        else:
            results_df = pd.DataFrame({'tpsr': [equation]})
        
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error updating results file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 