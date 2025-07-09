import pandas as pd
import os

def generate_leading_ones_ground_truth(instance_sizes=None):
    """
    Generate ground truth data for LeadingOnes problem.
    
    Creates a CSV file with columns:
    - InstanceSize: The size of the problem instance (10, 20, 30, 40, 50, 100)
    - CurrentState: All possible states from 0 to InstanceSize
    - Bitflip: Calculated as (instance_size/ (current_state + 1))
    
    Args:
        instance_sizes (list, optional): List of instance sizes to generate data for.
                                        If None, uses default sizes [10, 20, 30, 40, 50, 100]
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
            # Calculate bitflip value
            bitflip = (instance_size/ (current_state + 1))
            
            # Append data to lists
            instance_size_data.append(instance_size)
            current_state_data.append(current_state)
            bitflip_data.append(bitflip)
    
    # Create DataFrame
    df = pd.DataFrame({
        'InstanceSize': instance_size_data,
        'CurrentState': current_state_data,
        'Bitflip': bitflip_data
    })
    
    # Create Ground_Truth directory if it doesn't exist
    ground_truth_dir = '../Ground_Truth'
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    # Save to CSV file without headers
    output_path = os.path.join(ground_truth_dir, 'GTLeadingOnes.csv')
    df.to_csv(output_path, index=False, header=False)
    
    print(f"Ground truth data generated successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Instance sizes: {instance_sizes}")
    print(f"Sample data:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    import sys
    
    # Check if instance sizes are provided as command line arguments
    if len(sys.argv) > 1:
        try:
            # Parse instance sizes from command line arguments
            instance_sizes = [int(size) for size in sys.argv[1:]]
            print(f"Generating data for instance sizes: {instance_sizes}")
        except ValueError:
            print("Error: All arguments must be integers (instance sizes)")
            sys.exit(1)
    else:
        instance_sizes = None
        print("Using default instance sizes: [10, 20, 30, 40, 50, 100]")
    
    # Generate the ground truth data
    df = generate_leading_ones_ground_truth(instance_sizes) 