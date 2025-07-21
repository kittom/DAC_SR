import pandas as pd
import numpy as np
import os
import sys
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_one_max_ground_truth_with_noise(instance_sizes=None, noise_level=0.1, noise_type='gaussian', data_type='continuous'):
    """
    Generate ground truth data for OneMax problem with added noise.
    
    Creates a CSV file with columns:
    - InstanceSize: The size of the problem instance
    - CurrentState: All possible states from 0 to InstanceSize
    - Bitflip: Calculated as sqrt(instance_size/(instance_size-current_state)) + noise
    
    Args:
        instance_sizes (list, optional): List of instance sizes to generate data for.
                                        If None, uses default sizes [10, 20, 30, 40, 50, 100]
        noise_level (float): Standard deviation of the noise (default: 0.1)
        noise_type (str): Type of noise - 'gaussian' or 'uniform' (default: 'gaussian')
        data_type (str): Type of data to generate - 'continuous' or 'discrete'
    """
    
    # Define the instance sizes to cycle through
    if instance_sizes is None:
        instance_sizes = [10, 20, 30, 40, 50, 100]
    
    # Lists to store the data
    instance_size_data = []
    current_state_data = []
    bitflip_data = []
    
    # Generate data for each instance size
    for instance_size in instance_sizes:
        # For each instance size, generate all possible current states
        for current_state in range(instance_size + 1):
            # Calculate theoretical bitflip value for OneMax
            if current_state == instance_size:
                # Replace infinite values with a large finite value for compatibility
                theoretical_bitflip = 1000.0
            else:
                theoretical_bitflip = math.sqrt(instance_size / (instance_size - current_state))
            
            # Add noise based on the specified type
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_level, noise_level)
            else:
                raise ValueError("noise_type must be 'gaussian' or 'uniform'")
            
            # Calculate noisy bitflip value
            noisy_bitflip = theoretical_bitflip + noise
            
            # Ensure bitflip values are positive (noise might make them negative)
            noisy_bitflip = max(0.001, noisy_bitflip)
            
            # Round to nearest integer if discrete data is requested
            if data_type == 'discrete':
                noisy_bitflip = round(noisy_bitflip)
            
            # Append data to lists
            instance_size_data.append(instance_size)
            current_state_data.append(current_state)
            bitflip_data.append(noisy_bitflip)
    
    # Create DataFrame (without theoretical bitflip column)
    df = pd.DataFrame({
        'InstanceSize': instance_size_data,
        'CurrentState': current_state_data,
        'Bitflip': bitflip_data
    })
    
    # Create Ground_Truth_Noise/OneMax/{data_type} directory if it doesn't exist
    out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth_Noise/OneMax/{data_type}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Save to CSV file without headers, with noise level in filename
    output_filename = f'GTOneMax_Noise_{noise_type}_{noise_level}.csv'
    out_path = os.path.join(out_dir, output_filename)
    df.to_csv(out_path, index=False, header=False)
    
    print(f"Ground truth data with noise generated successfully!")
    print(f"Data type: {data_type}, Noise type: {noise_type}, Noise level: {noise_level}")
    print(f"Output saved to: {out_path}")
    print(f"Total rows: {len(df)}")
    print(f"Instance sizes: {instance_sizes}")
    print(f"Sample data (first 10 rows):")
    print(df.head(10))
    
    # Print noise statistics (calculate from the original theoretical values)
    noise_values = []
    for i, row in df.iterrows():
        if row['CurrentState'] == row['InstanceSize']:
            theoretical = 1000.0  # Use the same large finite value
        else:
            theoretical = math.sqrt(row['InstanceSize'] / (row['InstanceSize'] - row['CurrentState']))
        
        noise_values.append(row['Bitflip'] - theoretical)
    
    if noise_values:
        noise_values = np.array(noise_values)
        print(f"\nNoise Statistics:")
        print(f"Mean noise: {noise_values.mean():.6f}")
        print(f"Std noise: {noise_values.std():.6f}")
        print(f"Min noise: {noise_values.min():.6f}")
        print(f"Max noise: {noise_values.max():.6f}")
    else:
        print(f"\nNoise Statistics: No valid noise values to calculate")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate OneMax ground truth data with noise')
    parser.add_argument('instance_sizes', nargs='*', type=int, 
                       help='Instance sizes to generate data for (default: 10 20 30 40 50 100)')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level (standard deviation for gaussian, range for uniform) (default: 0.1)')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian',
                       help='Type of noise to add (default: gaussian)')
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous',
                       help='Type of data to generate (default: continuous)')
    
    args = parser.parse_args()
    
    # Use provided instance sizes or default
    if args.instance_sizes:
        instance_sizes = args.instance_sizes
        print(f"Generating data for instance sizes: {instance_sizes}")
    else:
        instance_sizes = None
        print("Using default instance sizes: [10, 20, 30, 40, 50, 100]")
    
    print(f"Data type: {args.data_type}")
    
    # Generate the ground truth data with noise
    df = generate_one_max_ground_truth_with_noise(
        instance_sizes=instance_sizes,
        noise_level=args.noise_level,
        noise_type=args.noise_type,
        data_type=args.data_type
    ) 