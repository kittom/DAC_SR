import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_leading_ones_ground_truth_with_noise(instance_sizes=None, noise_level=0.1, noise_type='gaussian'):
    if instance_sizes is None:
        instance_sizes = [10, 20, 30, 40, 50, 100]
    instance_size_data = []
    current_state_data = []
    bitflip_data = []
    for instance_size in instance_sizes:
        for current_state in range(instance_size + 1):
            theoretical_bitflip = (instance_size / (current_state + 1))
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_level, noise_level)
            else:
                raise ValueError("noise_type must be 'gaussian' or 'uniform'")
            noisy_bitflip = theoretical_bitflip + noise
            noisy_bitflip = max(0.001, noisy_bitflip)
            instance_size_data.append(instance_size)
            current_state_data.append(current_state)
            bitflip_data.append(noisy_bitflip)
    df = pd.DataFrame({
        'InstanceSize': instance_size_data,
        'CurrentState': current_state_data,
        'Bitflip': bitflip_data
    })
    return df

def write_results_files(output_dir, ground_truth_formula):
    os.makedirs(output_dir, exist_ok=True)
    # results.csv
    results_path = os.path.join(output_dir, 'results.csv')
    pd.DataFrame({'ground_truth': [ground_truth_formula]}).to_csv(results_path, index=False)
    # results_lib.csv
    results_lib_path = os.path.join(output_dir, 'results_lib.csv')
    pd.DataFrame({'ground_truth': [ground_truth_formula]}).to_csv(results_lib_path, index=False)

def write_results_rounding_file(output_dir, ground_truth_formula):
    os.makedirs(output_dir, exist_ok=True)
    results_rounding_path = os.path.join(output_dir, 'results_rounding.csv')
    pd.DataFrame({'ground_truth': [ground_truth_formula]}).to_csv(results_rounding_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate LeadingOnes ground truth data with noise')
    parser.add_argument('instance_sizes', nargs='*', type=int, 
                       help='Instance sizes to generate data for (default: 10 20 30 40 50 100)')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level (standard deviation for gaussian, range for uniform) (default: 0.1)')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian',
                       help='Type of noise to add (default: gaussian)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for the generated data')
    args = parser.parse_args()
    if args.instance_sizes:
        instance_sizes = args.instance_sizes
        print(f"Generating data for instance sizes: {instance_sizes}")
    else:
        instance_sizes = None
        print("Using default instance sizes: [10, 20, 30, 40, 50, 100]")
    print(f"Noise type: {args.noise_type}, Noise level: {args.noise_level}")
    if args.output_dir is not None:
        base_out_dir = args.output_dir
    else:
        base_out_dir = os.path.join(PROJECT_ROOT, 'DataSets/Ground_Truth_Noise/LeadingOnes')
    # Continuous
    cont_dir = os.path.join(base_out_dir, 'continuous')
    os.makedirs(cont_dir, exist_ok=True)
    df_cont = generate_leading_ones_ground_truth_with_noise(instance_sizes, args.noise_level, args.noise_type)
    cont_path = os.path.join(cont_dir, 'GTLeadingOnes.csv')
    df_cont.to_csv(cont_path, index=False, header=False)
    write_results_files(cont_dir, 'x1/(x2 + 1)')
    # Discrete (rounded)
    disc_dir = os.path.join(base_out_dir, 'discrete')
    os.makedirs(disc_dir, exist_ok=True)
    df_disc = df_cont.copy()
    df_disc["Bitflip"] = df_disc["Bitflip"].round().astype(int)
    disc_path = os.path.join(disc_dir, 'GTLeadingOnes.csv')
    df_disc.to_csv(disc_path, index=False, header=False)
    write_results_rounding_file(disc_dir, 'round(x1/(x2 + 1))')
    print(f"Continuous and discrete noisy datasets and results files written to {base_out_dir}") 