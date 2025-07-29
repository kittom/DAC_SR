import os
import sys
import csv
import argparse
import numpy as np
import ioh

# Add local modcma package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modcma'))
from modularcmaes import ModularCMAES

BENCHMARKS = [
    ("sphere", 1),
    ("ellipsoid", 2),
    ("rastrigin", 15),
    ("noisy_ellipsoid", 21),
    ("schaffer", 19),
    ("noisy_rastrigin", 23),
]

# Output root for all CSVs
OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../DataSets/Ground_Truth/PSACMAES'))

def run_psa_cmaes_benchmark(benchmark_name, fid, dim, budget_factor, restarts, output_dir, 
                           data_type='continuous', evaluation_type=None, hidden_variables=False, 
                           noise_level=0.0, noise_type='gaussian', use_rounded_values=False,
                           remove_alpha_beta=False, add_noisy_parameters=0):
    """
    Run PSA-CMA-ES benchmark with specified parameters.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'sphere')
        fid: Function ID for the benchmark
        dim: Problem dimension
        budget_factor: Budget scaling factor
        restarts: Number of independent restarts
        output_dir: Output directory for results
        data_type: 'continuous' or 'discrete'
        evaluation_type: 'control', 'library', 'rounding', or None
        hidden_variables: Whether to hide psa_beta and alpha from output
        noise_level: Level of noise to add (0.0 for no noise)
        noise_type: 'gaussian' or 'uniform' noise
    """
    algo = 'psa'
    function_id = fid
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine CSV columns based on settings
    if remove_alpha_beta:
        # Remove alpha and beta columns
        csv_cols = ['lambda', 'ptnorm', 'gamma_theta', 'next_lambda_unrounded']
        ground_truth_equation = 'x1 * exp(HIDDEN_ALPHA * (x3 - (x2^2 / HIDDEN_BETA)))'
    elif hidden_variables:
        csv_cols = ['lambda', 'ptnorm', 'gamma_theta', 'next_lambda_unrounded']
        ground_truth_equation = 'x1 * exp(HIDDEN_ALPHA * (x3 - (x2^2 / HIDDEN_BETA)))'
    else:
        csv_cols = ['lambda', 'psa_beta', 'ptnorm', 'alpha', 'gamma_theta', 'next_lambda_unrounded']
        ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3^2 / x4)))'
    
    # Add noisy parameter columns if requested
    if add_noisy_parameters > 0:
        for i in range(add_noisy_parameters):
            csv_cols.insert(-1, f'noisy_param_{i+1}')
    
    # Run multiple restarts and combine all data
    all_restart_data = []
    
    for restart in range(restarts):
        problem = ioh.get_problem(
            fid=function_id,
            instance=1,
            dimension=dim,
            problem_class=ioh.ProblemClass.BBOB
        )
        
        csv_path = os.path.join(output_dir, 'psa_vars.csv')
        
        try:
            cma = ModularCMAES(
                problem,
                dim,
                budget=dim * budget_factor,
                pop_size_adaptation='psa',
                min_lambda_=4,
                max_lambda_=512,
                x0=np.zeros((dim, 1)),
                lb=np.full((dim, 1), -5),
                ub=np.full((dim, 1), 5),
            )
            
            # Write CSV header only for the first restart
            write_header = (restart == 0)
            
            with open(csv_path, 'a' if restart > 0 else 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header only for the first restart
                if write_header:
                    writer.writerow(csv_cols)
                
                # Initialize gamma_theta
                gamma_theta = 0.0
                
                # Run until budget exhaustion
                while cma.parameters.used_budget < cma.parameters.budget:
                    # Record current state BEFORE step (so we get the current lambda before it gets updated)
                    lambda_ = cma.parameters.lambda_
                    psa_beta = cma.parameters.psa_beta
                    ptnorm = cma.parameters.ptnorm
                    alpha = cma.parameters.alpha
                    
                    # Use the modular CMA-ES's internal gamma_theta (which is the correct one for this generation)
                    gamma_theta = cma.parameters.gamma_theta
                    
                    # Now step the algorithm (which will update lambda_ for the next iteration)
                    cma.step()
                    
                    # Get the new lambda that the modular CMA-ES actually set
                    new_lambda = cma.parameters.lambda_
                    
                    # The modular CMA-ES calculated: new_lambda = lambda_ * exp(psa_beta * (gamma_theta - (ptnorm**2 / alpha)))
                    # We need to calculate the unrounded value that the modular CMA-ES should have used
                    
                    if use_rounded_values:
                        # Use the actual rounded value that was applied
                        next_lambda_unrounded = new_lambda
                    else:
                        # Use the unrounded value that the modular CMA-ES actually calculated
                        next_lambda_unrounded = cma.parameters.unrounded_lambda
                    
                    # Add noise if specified
                    if noise_level > 0.0:
                        if noise_type == 'gaussian':
                            noise = np.random.normal(0, noise_level)
                        elif noise_type == 'uniform':
                            noise = np.random.uniform(-noise_level, noise_level)
                        else:
                            raise ValueError("noise_type must be 'gaussian' or 'uniform'")
                        next_lambda_unrounded += noise
                    
                    # Round the value if discrete data is requested
                    # Note: We keep the unrounded value in the final column for continuous data
                    # The rounding only affects the actual population size used by the algorithm
                    if data_type == 'discrete':
                        next_lambda_unrounded = round(next_lambda_unrounded)
                    
                    # Write row based on settings
                    if remove_alpha_beta:
                        row = [lambda_, ptnorm, gamma_theta, next_lambda_unrounded]
                    elif hidden_variables:
                        row = [lambda_, ptnorm, gamma_theta, next_lambda_unrounded]
                    else:
                        row = [lambda_, psa_beta, ptnorm, alpha, gamma_theta, next_lambda_unrounded]
                    
                    # Add noisy parameters if requested
                    if add_noisy_parameters > 0:
                        # Calculate min and max values from the existing data for noise range
                        existing_values = [lambda_, psa_beta, ptnorm, alpha, gamma_theta, next_lambda_unrounded]
                        min_val = min(v for v in existing_values if isinstance(v, (int, float)))
                        max_val = max(v for v in existing_values if isinstance(v, (int, float)))
                        
                        # Generate noisy parameters within the range of existing data
                        for i in range(add_noisy_parameters):
                            noisy_val = np.random.uniform(min_val, max_val)
                            row.insert(-1, noisy_val)
                    
                    writer.writerow(row)
                    all_restart_data.append(row)
            
            print(f"{benchmark_name}_{dim}D restart_{restart + 1}: best y={problem.state.current_best.y}")
            problem.reset()
            
        except Exception as e:
            print(f"Error writing CSV for {benchmark_name}_{dim}D restart_{restart + 1}: {e}")
    
    # Write results files based on evaluation type with ground truth as first column (only once after all restarts)
    if evaluation_type == 'control' or (evaluation_type is None and data_type == 'continuous'):
        results_path = os.path.join(output_dir, 'results.csv')
    elif evaluation_type == 'library':
        results_path = os.path.join(output_dir, 'results_lib.csv')
    elif evaluation_type == 'rounding' or (evaluation_type is None and data_type == 'discrete'):
        results_path = os.path.join(output_dir, 'results_rounding.csv')
    else:
        pass  # No results file needed
        
    if 'results_path' in locals():
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
    
    return all_restart_data

def aggregate_csvs(benchmark_csvs, output_path, data_type='continuous', evaluation_type=None, 
                  hidden_variables=False, noise_level=0.0, noise_type='gaussian'):
    """Aggregate data from multiple benchmarks into a single CSV."""
    rows = []
    for benchmark_name, csv_data in benchmark_csvs:
        for row in csv_data:
            rows.append(row)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    # Write appropriate results file based on evaluation type
    if evaluation_type == 'control' or (evaluation_type is None and data_type == 'continuous'):
        results_path = output_path.replace('.csv', '_results.csv')
    elif evaluation_type == 'library':
        results_path = output_path.replace('.csv', '_results_lib.csv')
    elif evaluation_type == 'rounding' or (evaluation_type is None and data_type == 'discrete'):
        results_path = output_path.replace('.csv', '_results_rounding.csv')
    else:
        return
        
    # Determine ground truth equation based on hidden_variables setting
    if hidden_variables:
        ground_truth_equation = 'x1 * exp(HIDDEN_ALPHA * (x3 - (x2^2 / HIDDEN_BETA)))'
    else:
        ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3^2 / x4)))'
    
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ground_truth'])
        writer.writerow([ground_truth_equation])

def create_leave_one_out_datasets(benchmark_csvs, output_root, data_type='continuous', 
                                evaluation_type=None, hidden_variables=False, noise_level=0.0, noise_type='gaussian'):
    """Create leave-one-out comparison datasets."""
    for excluded_name, excluded_data in benchmark_csvs:
        excluded_dir = os.path.join(output_root, f'absent_{excluded_name}')
        os.makedirs(excluded_dir, exist_ok=True)
        
        combined_rows = []
        for benchmark_name, csv_data in benchmark_csvs:
            if benchmark_name != excluded_name:
                combined_rows.extend(csv_data)
        
        combined_csv_path = os.path.join(excluded_dir, 'psa_vars.csv')
        with open(combined_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(combined_rows)
        
        # Write appropriate results file based on evaluation type
        if evaluation_type == 'control' or (evaluation_type is None and data_type == 'continuous'):
            results_path = os.path.join(excluded_dir, 'results.csv')
        elif evaluation_type == 'library':
            results_path = os.path.join(excluded_dir, 'results_lib.csv')
        elif evaluation_type == 'rounding' or (evaluation_type is None and data_type == 'discrete'):
            results_path = os.path.join(excluded_dir, 'results_rounding.csv')
        else:
            continue
            
        # Determine ground truth equation based on hidden_variables setting
        if hidden_variables:
            ground_truth_equation = 'x1 * exp(HIDDEN_ALPHA * (x3 - (x2 / HIDDEN_BETA)))'
        else:
            ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3 / x4)))'
        
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
        
        print(f"Created leave-one-out dataset excluding {excluded_name}: {combined_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate PSA-CMA-ES data with comprehensive configuration options.")
    
    # Core parameters
    parser.add_argument('--budget-factor', type=int, required=True, help='Budget scaling factor (e.g., 500)')
    parser.add_argument('--restarts', type=int, default=5, help='Number of independent restarts (default: 5)')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[2, 3], help='Problem dimensions (default: 2 3)')
    
    # Data generation parameters
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous', help='Type of data to generate')
    parser.add_argument('--evaluation-type', choices=['control', 'library', 'rounding'], default=None, help='Type of evaluation to generate results for')
    
    # Feature flags
    parser.add_argument('--hidden-variables', action='store_true', help='Hide psa_beta and alpha from output')
    parser.add_argument('--noise-level', type=float, default=0.0, help='Level of noise to add (default: 0.0)')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian', help='Type of noise to add')
    parser.add_argument('--use-rounded-values', action='store_true', help='Use rounded values in the final column instead of unrounded')
    parser.add_argument('--remove-alpha-beta', action='store_true', help='Remove alpha and beta columns from output')
    parser.add_argument('--add-noisy-parameters', type=int, default=0, help='Number of noisy parameters to add (default: 0)')
    
    # Output and organization
    parser.add_argument('--output-root', type=str, default=None, help='Optional output root directory for CSVs')
    parser.add_argument('--compare', action='store_true', help='Create leave-one-out comparison datasets')
    parser.add_argument('--individual-benchmarks', type=str, choices=['true', 'false'], default='true', help='Generate individual benchmark datasets (true/false)')
    parser.add_argument('--all-benchmarks', type=str, choices=['true', 'false'], default='true', help='Generate all_benchmarks.csv aggregated dataset (true/false)')
    parser.add_argument('--sub-benchmarks', nargs='+', default=None, help='List of sub-benchmarks to generate')
    
    args = parser.parse_args()
    
    # Parse boolean arguments
    individual_benchmarks = args.individual_benchmarks.lower() == 'true'
    all_benchmarks = args.all_benchmarks.lower() == 'true'
    
    # Use specified sub-benchmarks or default to all
    if args.sub_benchmarks:
        filtered_benchmarks = []
        for benchmark_name, fid in BENCHMARKS:
            if benchmark_name in args.sub_benchmarks:
                filtered_benchmarks.append((benchmark_name, fid))
        benchmarks_to_generate = filtered_benchmarks
    else:
        benchmarks_to_generate = BENCHMARKS
    
    # Set output root
    if args.output_root is not None:
        output_root = os.path.abspath(args.output_root)
    else:
        output_root = os.path.join(OUTPUT_ROOT, f'budget_{args.budget_factor}_restarts_{args.restarts}')
    
    # Generate individual benchmark data
    benchmark_csvs = []
    temp_dirs = []
    
    if individual_benchmarks:
        # Generate individual benchmarks in the main output directory
        for benchmark_name, fid in benchmarks_to_generate:
            for dim in args.dimensions:
                output_dir = os.path.join(output_root, f'{benchmark_name}_{dim}D')
                csv_data = run_psa_cmaes_benchmark(
                    benchmark_name, fid, dim, args.budget_factor, args.restarts, output_dir,
                    args.data_type, args.evaluation_type, args.hidden_variables,
                    args.noise_level, args.noise_type, use_rounded_values=args.use_rounded_values,
                    remove_alpha_beta=args.remove_alpha_beta, add_noisy_parameters=args.add_noisy_parameters
                )
                benchmark_csvs.append((f'{benchmark_name}_{dim}D', csv_data))
        print(f"Generated individual benchmark datasets: {[name for name, _ in benchmark_csvs]}")
    else:
        # Generate individual benchmarks in a temporary directory
        import tempfile
        temp_root = tempfile.mkdtemp(prefix="psa_temp_")
        print(f"Generating temporary individual benchmark datasets in: {temp_root}")
        
        for benchmark_name, fid in benchmarks_to_generate:
            for dim in args.dimensions:
                output_dir = os.path.join(temp_root, f'{benchmark_name}_{dim}D')
                csv_data = run_psa_cmaes_benchmark(
                    benchmark_name, fid, dim, args.budget_factor, args.restarts, output_dir,
                    args.data_type, args.evaluation_type, args.hidden_variables,
                    args.noise_level, args.noise_type, use_rounded_values=args.use_rounded_values,
                    remove_alpha_beta=args.remove_alpha_beta, add_noisy_parameters=args.add_noisy_parameters
                )
                benchmark_csvs.append((f'{benchmark_name}_{dim}D', csv_data))
                temp_dirs.append(output_dir)
        print(f"Generated temporary individual benchmark datasets: {[name for name, _ in benchmark_csvs]}")
    
    # Generate all_benchmarks.csv if enabled
    if all_benchmarks and benchmark_csvs:
        aggregate_csvs(benchmark_csvs, os.path.join(output_root, 'all_benchmarks.csv'), 
                      args.data_type, args.evaluation_type, args.hidden_variables,
                      args.noise_level, args.noise_type)
        print(f"Aggregated CSV written to {os.path.join(output_root, 'all_benchmarks.csv')}")
    
    # Create leave-one-out datasets if compare is enabled
    if args.compare and benchmark_csvs:
        create_leave_one_out_datasets(benchmark_csvs, output_root, args.data_type, 
                                    args.evaluation_type, args.hidden_variables,
                                    args.noise_level, args.noise_type)
        print("Leave-one-out comparison datasets created successfully")
    
    # Clean up temporary directories if they were created
    if not individual_benchmarks and temp_dirs:
        import shutil
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)
        print("Cleaned up temporary individual benchmark datasets")

if __name__ == "__main__":
    main() 