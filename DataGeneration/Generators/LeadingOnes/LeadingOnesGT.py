import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_leading_ones_ground_truth(instance_sizes=None, data_type='continuous', output_dir=None):
    """
    Generate ground truth data for LeadingOnes problem.
    """
    if instance_sizes is None:
        instance_sizes = [10, 20, 30, 40, 50, 100]
    instance_size_data = []
    current_state_data = []
    bitflip_data = []
    for instance_size in instance_sizes:
        for current_state in range(instance_size + 1):
            bitflip = (instance_size/ (current_state + 1))
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
        out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/LeadingOnes/{data_type}')
    else:
        out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'GTLeadingOnes.csv')
    df.to_csv(out_path, index=False, header=False)
    print(f"Output saved to: {out_path}")
    
    # Create results.csv with ground truth equation
    ground_truth_equation = "x1/(x2 + 1)" if data_type == 'continuous' else "round(x1/(x2 + 1))"
    results_filename = 'results_rounding.csv' if data_type == 'discrete' else 'results.csv'
    results_path = os.path.join(out_dir, results_filename)
    
    # Create results DataFrame with ground truth as first column
    results_data = {
        'ground_truth': [ground_truth_equation]
    }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_path, index=False)
    print(f"Ground truth results saved to: {results_path}")
    print(f"Ground truth equation: {ground_truth_equation}")
    
    return df

if __name__ == "__main__":
    instance_sizes = None
    data_type = 'continuous'
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--data-type':
            if i + 1 < len(sys.argv):
                data_type = sys.argv[i + 1]
                if data_type not in ['continuous', 'discrete']:
                    print("Error: data_type must be 'continuous' or 'discrete'")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --data-type requires a value")
                sys.exit(1)
        else:
            try:
                if instance_sizes is None:
                    instance_sizes = []
                instance_sizes.append(int(sys.argv[i]))
                i += 1
            except ValueError:
                print(f"Warning: Ignoring non-numeric argument: {sys.argv[i]}")
                i += 1
    output_dir = None
    # Parse --output-dir argument
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
            i = idx + 2
        else:
            print("Error: --output-dir requires a value")
            sys.exit(1)
    df = generate_leading_ones_ground_truth(instance_sizes, data_type, output_dir) 