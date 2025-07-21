import pandas as pd
import os
import sys
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_one_max_ground_truth(instance_sizes=None, data_type='continuous'):
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
    out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/OneMax/{data_type}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'GTOneMax.csv')
    df.to_csv(out_path, index=False, header=False)
    print(f"Output saved to: {out_path}")
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
    df = generate_one_max_ground_truth(instance_sizes, data_type) 