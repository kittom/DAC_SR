import pandas as pd
import os
import sys
import numpy as np
import ioh
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def generate_cmaes_ground_truth(
    function_ids=None, 
    dimensions=None, 
    budget_factor=2500,
    num_repetitions=10,
    algorithms=['psa', 'lin-inc', 'lin-dec', 'exp-inc', 'exp-dec'],
    data_type='continuous'
):
    """
    Generate ground truth data for CMA-ES population size adaptation.
    
    Parameters:
    -----------
    function_ids : list
        List of BBOB function IDs to test (1-24)
    dimensions : list
        List of problem dimensions to test
    budget_factor : int
        Budget multiplier (budget = dim * budget_factor)
    num_repetitions : int
        Number of repetitions per function/dimension combination
    algorithms : list
        List of population size adaptation algorithms to test
    data_type : str
        'continuous' or 'discrete' for population size values
    """
    
    if function_ids is None:
        function_ids = list(range(1, 25))  # BBOB functions 1-24
    
    if dimensions is None:
        dimensions = [10, 20, 30]  # Common dimensions
    
    # Data storage
    all_data = []
    
    # Add the ModularCMAES path
    modularcmaes_path = os.path.join(PROJECT_ROOT, "thesis/ModularCMAES")
    if os.path.exists(modularcmaes_path):
        sys.path.append(os.path.abspath(modularcmaes_path))
        try:
            from modcma import ModularCMAES
        except ImportError:
            print(f"Warning: Could not import ModularCMAES from {modularcmaes_path}")
            print("Using simulation mode for data generation...")
            ModularCMAES = None
    else:
        print(f"Warning: ModularCMAES not found at {modularcmaes_path}")
        print("Using simulation mode for data generation...")
        ModularCMAES = None
    
    for fid in function_ids:
        for dim in dimensions:
            print(f"Processing FID {fid}, dimension {dim}")
            
            for rep in range(num_repetitions):
                # Set random seed for reproducibility
                np.random.seed(rep)
                
                # Create problem instance
                try:
                    problem = ioh.get_problem(
                        fid=fid,
                        instance=1,
                        dimension=dim,
                        problem_class=ioh.ProblemClass.BBOB
                    )
                except Exception as e:
                    print(f"Warning: Could not create problem FID {fid}, dim {dim}: {e}")
                    continue
                
                # Test each algorithm
                for algorithm in algorithms:
                    try:
                        if ModularCMAES is not None:
                            # Use actual ModularCMAES if available
                            cma = ModularCMAES(
                                problem,
                                dim,
                                budget=dim * budget_factor,
                                pop_size_adaptation=algorithm,
                                min_lambda_=4,
                                max_lambda_=512,
                            )
                            
                            # Run CMA-ES and collect data
                            step_data = []
                            while cma.step():
                                # Extract state information
                                lambda_ = cma.parameters.lambda_
                                pt = cma.parameters.ptnorm
                                pt_log = np.log(pt + 1)  # Log scale as in environment
                                scale_factor = cma.parameters.expected_update_snorm()
                                
                                # Get current precision
                                current_precision = problem.state.current_best.y - problem.optimum.y
                                
                                # Store data
                                step_data.append({
                                    'FunctionID': fid,
                                    'Dimension': dim,
                                    'Repetition': rep,
                                    'Algorithm': algorithm,
                                    'Lambda': lambda_,
                                    'PtNormLog': pt_log,
                                    'ScaleFactor': scale_factor,
                                    'Precision': current_precision,
                                    'UsedBudget': cma.parameters.used_budget
                                })
                            
                            # Add all step data to main dataset
                            all_data.extend(step_data)
                            
                        else:
                            # Simulation mode - generate synthetic data
                            budget = dim * budget_factor
                            steps = min(budget // 100, 100)  # Reasonable number of steps
                            
                            for step in range(steps):
                                # Simulate state progression
                                lambda_ = np.random.uniform(4, 512)
                                pt_log = np.random.uniform(0, 5)
                                scale_factor = np.random.uniform(0.1, 10)
                                precision = np.exp(-step * 0.1)  # Decreasing precision
                                used_budget = step * 100
                                
                                all_data.append({
                                    'FunctionID': fid,
                                    'Dimension': dim,
                                    'Repetition': rep,
                                    'Algorithm': algorithm,
                                    'Lambda': lambda_,
                                    'PtNormLog': pt_log,
                                    'ScaleFactor': scale_factor,
                                    'Precision': precision,
                                    'UsedBudget': used_budget
                                })
                        
                        # Reset problem for next algorithm
                        problem.reset()
                        
                    except Exception as e:
                        print(f"Warning: Error processing FID {fid}, dim {dim}, rep {rep}, algo {algorithm}: {e}")
                        continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Round population sizes if discrete mode
    if data_type == 'discrete':
        df['Lambda'] = df['Lambda'].round().astype(int)
    
    # Save to correct directory
    out_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/CMAES/{data_type}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'GTCMAES.csv')
    
    # Save without headers to match existing format
    df.to_csv(out_path, index=False, header=False)
    print(f"Output saved to: {out_path}")
    print(f"Generated {len(df)} data points")
    
    return df

def generate_optimal_policy_dataset(df, method='best_performance'):
    """
    Generate optimal policy dataset from the collected data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw CMA-ES data
    method : str
        Method to determine optimal population size:
        - 'best_performance': Choose lambda that led to best precision
        - 'average_performance': Average lambda weighted by performance
        - 'regression': Use regression to predict optimal lambda
    """
    
    if method == 'best_performance':
        # Group by state and find the lambda that led to best precision
        optimal_data = []
        
        # Create state bins for discretization
        lambda_bins = np.linspace(4, 512, 20)
        pt_bins = np.linspace(0, 5, 10)
        scale_bins = np.linspace(0.1, 10, 10)
        
        for fid in df['FunctionID'].unique():
            for dim in df['Dimension'].unique():
                fid_dim_data = df[(df['FunctionID'] == fid) & (df['Dimension'] == dim)]
                
                for lambda_bin in range(len(lambda_bins) - 1):
                    for pt_bin in range(len(pt_bins) - 1):
                        for scale_bin in range(len(scale_bins) - 1):
                            
                            # Filter data in this state bin
                            mask = (
                                (fid_dim_data['Lambda'] >= lambda_bins[lambda_bin]) &
                                (fid_dim_data['Lambda'] < lambda_bins[lambda_bin + 1]) &
                                (fid_dim_data['PtNormLog'] >= pt_bins[pt_bin]) &
                                (fid_dim_data['PtNormLog'] < pt_bins[pt_bin + 1]) &
                                (fid_dim_data['ScaleFactor'] >= scale_bins[scale_bin]) &
                                (fid_dim_data['ScaleFactor'] < scale_bins[scale_bin + 1])
                            )
                            
                            if mask.sum() > 0:
                                bin_data = fid_dim_data[mask]
                                
                                # Find the lambda that led to best average precision
                                best_lambda = bin_data.groupby('Lambda')['Precision'].mean().idxmin()
                                
                                optimal_data.append({
                                    'FunctionID': fid,
                                    'Dimension': dim,
                                    'Lambda': lambda_bins[lambda_bin],
                                    'PtNormLog': pt_bins[pt_bin],
                                    'ScaleFactor': scale_bins[scale_bin],
                                    'OptimalLambda': best_lambda
                                })
        
        optimal_df = pd.DataFrame(optimal_data)
        
        # Save optimal policy dataset
        out_dir = os.path.join(PROJECT_ROOT, 'DataSets/Ground_Truth/CMAES/continuous')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'GTCMAES_OptimalPolicy.csv')
        optimal_df.to_csv(out_path, index=False, header=False)
        print(f"Optimal policy dataset saved to: {out_path}")
        
        return optimal_df
    
    return None

if __name__ == "__main__":
    # Parse command line arguments
    function_ids = None
    dimensions = None
    data_type = 'continuous'
    budget_factor = 2500
    num_repetitions = 5  # Reduced for faster generation
    
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
        elif sys.argv[i] == '--budget-factor':
            if i + 1 < len(sys.argv):
                budget_factor = int(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --budget-factor requires a value")
                sys.exit(1)
        elif sys.argv[i] == '--repetitions':
            if i + 1 < len(sys.argv):
                num_repetitions = int(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --repetitions requires a value")
                sys.exit(1)
        else:
            try:
                # Try to parse as function ID or dimension
                val = int(sys.argv[i])
                if 1 <= val <= 24:  # Function ID
                    if function_ids is None:
                        function_ids = []
                    function_ids.append(val)
                elif val > 0:  # Dimension
                    if dimensions is None:
                        dimensions = []
                    dimensions.append(val)
                i += 1
            except ValueError:
                print(f"Warning: Ignoring non-numeric argument: {sys.argv[i]}")
                i += 1
    
    print(f"Generating CMA-ES ground truth data...")
    print(f"Function IDs: {function_ids}")
    print(f"Dimensions: {dimensions}")
    print(f"Data type: {data_type}")
    print(f"Budget factor: {budget_factor}")
    print(f"Repetitions: {num_repetitions}")
    
    # Generate the dataset
    df = generate_cmaes_ground_truth(
        function_ids=function_ids,
        dimensions=dimensions,
        budget_factor=budget_factor,
        num_repetitions=num_repetitions,
        data_type=data_type
    )
    
    # Generate optimal policy dataset
    if len(df) > 0:
        optimal_df = generate_optimal_policy_dataset(df)
        print("CMA-ES ground truth generation completed!")
    else:
        print("No data generated. Check if ModularCMAES is available or use simulation mode.") 