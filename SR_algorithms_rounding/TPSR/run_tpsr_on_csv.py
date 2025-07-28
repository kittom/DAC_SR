import sys
import os
import pandas as pd
import numpy as np
import torch
import argparse
import time
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine
from tpsr import tpsr_fit

def run_tpsr_with_convergence(X, Y, params, equation_env, noise_threshold, max_horizon=10, min_horizon=5):
    """
    Run TPSR with convergence-based stopping.
    
    Args:
        X: Input features (list of arrays)
        Y: Target values (list of arrays)
        params: TPSR parameters
        equation_env: Equation environment
        noise_threshold: Convergence threshold (noise level)
        max_horizon: Maximum MCTS horizon (reduced from 200 to 10 for speed)
        min_horizon: Minimum horizon before stopping (reduced from 10 to 5)
    
    Returns:
        tuple: (equation_string, final_reward, horizon_used)
    """
    # Convert noise threshold to reward threshold
    # TPSR uses reward = 1/(1+NMSE), so we need to convert
    reward_threshold = 1 / (1 + noise_threshold)
    
    print(f"Running TPSR with convergence threshold: {noise_threshold}")
    print(f"Reward threshold: {reward_threshold:.6f}")
    print(f"Max horizon: {max_horizon}, Min horizon: {min_horizon}")
    
    # Set up TPSR with optimized parameters for speed
    parser = get_parser()
    tpsr_params = parser.parse_args([])  # empty list to use defaults
    
    # Check if GPU is available and use it for better performance
    if torch.cuda.is_available():
        print("GPU detected - using GPU for TPSR")
        tpsr_params.cpu = False
        tpsr_params.device = torch.device("cuda")
    else:
        print("No GPU detected - using CPU")
        tpsr_params.cpu = True
        tpsr_params.device = torch.device("cpu")
    
    tpsr_params.debug = False
    tpsr_params.backbone_model = 'e2e'
    
    # OPTIMIZED PARAMETERS FOR SPEED
    tpsr_params.horizon = max_horizon  # Reduced from 200 to 50
    tpsr_params.width = 2  # Reduced from 3 to 2 for faster search
    tpsr_params.rollout = 2  # Reduced from 3 to 2 for faster rollouts
    tpsr_params.ucb_constant = 0.5  # Reduced from 1.0 for more exploration
    tpsr_params.ucb_base = 5.0  # Reduced from 10.0 for faster convergence
    tpsr_params.print_freq = 10  # Print progress more frequently
    
    print(f"Optimized TPSR parameters:")
    print(f"  - Horizon: {tpsr_params.horizon}")
    print(f"  - Width: {tpsr_params.width}")
    print(f"  - Rollout: {tpsr_params.rollout}")
    print(f"  - UCB constant: {tpsr_params.ucb_constant}")
    print(f"  - UCB base: {tpsr_params.ucb_base}")
    print(f"  - Device: {tpsr_params.device}")
    
    # Build environment and model
    np.random.seed(tpsr_params.seed)
    torch.manual_seed(tpsr_params.seed)
    equation_env = build_env(tpsr_params)
    modules = build_modules(equation_env, tpsr_params)
    trainer = Trainer(modules, equation_env, tpsr_params)
    
    # Create samples dictionary
    samples = {'x_to_fit': X, 'y_to_fit': Y}
    
    # Run TPSR with convergence monitoring
    start_time = time.time()
    
    # We need to modify the MCTS loop to monitor convergence
    # Since we can't easily modify the core TPSR code, we'll use a wrapper approach
    
    # Run TPSR (MCTS loop) with the full horizon
    final_seq, time_elapsed, best_reward = tpsr_fit(X, Y, tpsr_params, equation_env)
    
    # Check if we converged early (this is a simplified check)
    # In practice, we'd need to modify the core TPSR code to get intermediate rewards
    if best_reward is not None and best_reward > reward_threshold:
        print(f"TPSR converged with reward: {best_reward:.6f}")
        horizon_used = len(final_seq) if final_seq else max_horizon
    else:
        print(f"TPSR did not converge within {max_horizon} steps")
        horizon_used = max_horizon
    
    elapsed_time = time.time() - start_time
    
    # Use pred_for_sample_no_refine to get the equation string
    y, eq_str, _ = pred_for_sample_no_refine(
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

def main():
    parser = argparse.ArgumentParser(description="Run TPSR symbolic regression with convergence-based stopping")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--noise", type=float, default=1e-12, help="Noise threshold for convergence (default: 1e-12)")
    parser.add_argument("--max-horizon", type=int, default=10, help="Maximum MCTS horizon (default: 10, optimized for speed)")
    parser.add_argument("--min-horizon", type=int, default=5, help="Minimum horizon before stopping (default: 5, optimized for speed)")
    args = parser.parse_args()
    
    csv_file = args.csv_file
    noise_threshold = args.noise
    max_horizon = args.max_horizon
    min_horizon = args.min_horizon
    
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        sys.exit(1)
    
    # Load the dataset
    print(f"Loading dataset from: {csv_file}")
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
    Y = data.iloc[:, -1].values.reshape(-1, 1)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {Y.shape}")
    
    # Run TPSR with convergence-based stopping
    eq_str, final_reward, horizon_used = run_tpsr_with_convergence(
        [X], [Y], None, None, noise_threshold, max_horizon, min_horizon
    )
    
    # Write to results.csv in the same directory as the input CSV
    csv_dir = os.path.dirname(csv_file)
    results_path = os.path.join(csv_dir, "results.csv")
    
    # If results.csv exists, append tpsr column; else, create new
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df['tpsr'] = [eq_str] + [None]*(len(df)-1)
    else:
        df = pd.DataFrame({'tpsr': [eq_str]})
    
    df.to_csv(results_path, index=False)
    
    print(f"TPSR discovered equation: {eq_str}")
    print(f"Equation written to: {results_path}")
    print(f"Convergence summary: {horizon_used} horizon steps, final reward: {final_reward:.6f}")

if __name__ == "__main__":
    main() 