import pandas as pd
import os
import sys
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_one_max_ground_truth(instance_sizes=None, data_type='continuous', output_dir=None, evaluation_type=None):
    """
    Generate ground truth data for OneMax problem.
    """
    if instance_sizes is None:
        instance_sizes = [10, 20, 30, 40, 50, 100]
    instance_size_data = []
    current_state_data = []
    bitflip_data = []
    for instance_size in instance_sizes:
        for current_state in range(instance_size + 1):
            # OneMax uses sqrt(instance_size/(instance_size-current_state))
            if current_state == instance_size:
                # Replace infinite values with a large finite value for compatibility
                bitflip = 0.0
            else:
                bitflip = math.sqrt(instance_size / (instance_size - current_state))
            if data_type == 'discrete':
                bitflip = round(bitflip)
            instance_size_data.append(instance_size)
            current_state_data.append(current_state)
            bitflip_data.append(bitflip)
    df = pd.DataFrame({
        'InstanceSize': instance_size_data,
        'CurrentState': current_state_data,
        'Bitflip': bitflip_data
    })
    # Save to correct directory
    if output_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/OneMax/{data_type}')
    else:
        out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'GTOneMax.csv')
    df.to_csv(out_path, index=False, header=False)
    print(f"Output saved to: {out_path}")
    
    # Create ground truth equation
    ground_truth_equation = "sqrt(x1/(x1-x2))" if data_type == 'continuous' else "round(sqrt(x1/(x1-x2)))"
    
    if data_type == 'continuous':
        # Create results file based on evaluation type
        if evaluation_type == 'control' or evaluation_type is None:
            # For control library evaluation (results.csv)
            control_results_path = os.path.join(out_dir, 'results.csv')
            control_results_data = {
                'ground_truth': [ground_truth_equation]
            }
            control_results_df = pd.DataFrame(control_results_data)
            control_results_df.to_csv(control_results_path, index=False)
            print(f"Control library results file created: {control_results_path}")
        
        if evaluation_type == 'library' or evaluation_type is None:
            # For tailored library evaluation (results_lib.csv)
            tailored_results_path = os.path.join(out_dir, 'results_lib.csv')
            tailored_results_data = {
                'ground_truth': [ground_truth_equation]
            }
            tailored_results_df = pd.DataFrame(tailored_results_data)
            tailored_results_df.to_csv(tailored_results_path, index=False)
            print(f"Tailored library results file created: {tailored_results_path}")
        
    else:  # discrete
        # Create one results file for discrete data (rounding evaluation)
        rounding_results_path = os.path.join(out_dir, 'results_rounding.csv')
        rounding_results_data = {
            'ground_truth': [ground_truth_equation]
        }
        rounding_results_df = pd.DataFrame(rounding_results_data)
        rounding_results_df.to_csv(rounding_results_path, index=False)
        print(f"Rounding results file created: {rounding_results_path}")
    
    print(f"Ground truth equation: {ground_truth_equation}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate OneMax ground truth data")
    parser.add_argument('instance_sizes', type=int, nargs='*', default=[10, 20, 30, 40, 50, 100, 200, 500],
                       help='Instance sizes to generate data for')
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous',
                       help='Type of data to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for the generated data')
    parser.add_argument('--evaluation-type', choices=['control', 'library', 'rounding'], default=None,
                       help='Type of evaluation to prepare results for')
    
    args = parser.parse_args()
    
    df = generate_one_max_ground_truth(args.instance_sizes, args.data_type, args.output_dir, args.evaluation_type) 