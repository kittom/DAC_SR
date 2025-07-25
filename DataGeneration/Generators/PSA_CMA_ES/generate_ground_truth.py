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

# Only these columns (no generation, no benchmark)
CSV_COLS = ['lambda', 'psa_beta', 'ptnorm', 'alpha', 'gamma_theta', 'next_lambda_unrounded']


def run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir, data_type='continuous', evaluation_type=None):
    dim = 10
    budget_factor = 2500
    algo = 'psa'
    function_id = fid
    problem = ioh.get_problem(
        fid=function_id,
        instance=1,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'psa_vars.csv')
    ground_truth_path = os.path.join(output_dir, 'ground_truth.csv')
    ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3 / x4)))'
    try:
        cma = ModularCMAES(
            problem,
            dim,
            budget=dim*budget_factor,
            pop_size_adaptation='psa',
            min_lambda_=4,
            max_lambda_=512,
            x0=np.zeros((dim, 1)),
            lb=np.full((dim, 1), -5),
            ub=np.full((dim, 1), 5),
        )
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # No header row
            cma.step()
            lambda_ = cma.parameters.lambda_
            psa_beta = cma.parameters.psa_beta
            ptnorm = cma.parameters.ptnorm
            alpha = cma.parameters.alpha
            gamma_theta = 0.0  # Initialize gamma_theta to 0
            next_lambda_unrounded = lambda_ * np.exp(psa_beta * (gamma_theta - (ptnorm / alpha)))
            
            # Round the value if discrete data is requested
            if data_type == 'discrete':
                next_lambda_unrounded = round(next_lambda_unrounded)
                
            writer.writerow([
                lambda_, psa_beta, ptnorm, alpha, gamma_theta, next_lambda_unrounded
            ])
            for _ in range(iterations-1):
                cma.step()
                lambda_ = cma.parameters.lambda_
                psa_beta = cma.parameters.psa_beta
                ptnorm = cma.parameters.ptnorm
                alpha = cma.parameters.alpha
                # Update gamma_theta: gamma_theta = (1-psa_beta)^2 * gamma_theta + psa_beta * (2-psa_beta)
                gamma_theta = (1 - psa_beta)**2 * gamma_theta + psa_beta * (2 - psa_beta)
                next_lambda_unrounded = lambda_ * np.exp(psa_beta * (gamma_theta - (ptnorm / alpha)))
                
                # Round the value if discrete data is requested
                if data_type == 'discrete':
                    next_lambda_unrounded = round(next_lambda_unrounded)
                    
                writer.writerow([
                    lambda_, psa_beta, ptnorm, alpha, gamma_theta, next_lambda_unrounded
                ])
        # Write results files based on evaluation type with ground truth as first column
        if evaluation_type == 'control' or (evaluation_type is None and data_type == 'continuous'):
            # Create results.csv for control evaluation
            control_results_path = os.path.join(output_dir, 'results.csv')
            with open(control_results_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ground_truth'])
                writer.writerow([ground_truth_equation])
        elif evaluation_type == 'library':
            # Create results_lib.csv for library evaluation
            tailored_results_path = os.path.join(output_dir, 'results_lib.csv')
            with open(tailored_results_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ground_truth'])
                writer.writerow([ground_truth_equation])
        elif evaluation_type == 'rounding' or (evaluation_type is None and data_type == 'discrete'):
            # Create results_rounding.csv for rounding evaluation
            rounding_results_path = os.path.join(output_dir, 'results_rounding.csv')
            with open(rounding_results_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ground_truth'])
                writer.writerow([ground_truth_equation])
    except Exception as e:
        print(f"Error writing CSV for {benchmark_name}: {e}")
    print(f"{benchmark_name}: best y={problem.state.current_best.y}")
    problem.reset()
    return csv_path

def aggregate_csvs(benchmark_csvs, output_path, data_type='continuous', evaluation_type=None):
    rows = []
    for benchmark_name, csv_path in benchmark_csvs:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # row: [lambda, psa_beta, ptnorm, alpha, update_term, exp_update_value, next_lambda_unrounded]
                rows.append(row)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # No header row
        writer.writerows(rows)
    # Write results files for the aggregated file based on evaluation type
    ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3 / x4)))'
    
    if evaluation_type == 'control' or (evaluation_type is None and data_type == 'continuous'):
        # Create results.csv for control evaluation
        control_results_path = os.path.join(os.path.dirname(output_path), 'results.csv')
        with open(control_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
    elif evaluation_type == 'library':
        # Create results_lib.csv for library evaluation
        tailored_results_path = os.path.join(os.path.dirname(output_path), 'results_lib.csv')
        with open(tailored_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
    elif evaluation_type == 'rounding' or (evaluation_type is None and data_type == 'discrete'):
        # Create results_rounding.csv for rounding evaluation
        rounding_results_path = os.path.join(os.path.dirname(output_path), 'results_rounding.csv')
        with open(rounding_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])


def create_leave_one_out_datasets(benchmark_csvs, output_root, data_type='continuous', evaluation_type=None):
    """Create leave-one-out datasets for comparison experiments."""
    ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3 / x4)))'
    
    # Create absent directory
    absent_dir = os.path.join(output_root, 'absent')
    os.makedirs(absent_dir, exist_ok=True)
    
    # For each benchmark, create a dataset with all other benchmarks
    for excluded_benchmark, excluded_csv_path in benchmark_csvs:
        excluded_name = excluded_benchmark
        
        # Create subdirectory for this excluded benchmark
        excluded_dir = os.path.join(absent_dir, excluded_name)
        os.makedirs(excluded_dir, exist_ok=True)
        
        # Combine all other benchmarks
        combined_rows = []
        for benchmark_name, csv_path in benchmark_csvs:
            if benchmark_name != excluded_benchmark:
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        combined_rows.append(row)
        
        # Write combined dataset
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
            
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
        
        print(f"Created leave-one-out dataset excluding {excluded_name}: {combined_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate PSA-CMA-ES ground truth data for multiple benchmarks.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of generations to run for each benchmark')
    parser.add_argument('--output-root', type=str, default=None, help='Optional output root directory for CSVs')
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous', help='Type of data to generate')
    parser.add_argument('--evaluation-type', choices=['control', 'library', 'rounding'], default=None, help='Type of evaluation to generate results for')
    parser.add_argument('--compare', action='store_true', help='Create leave-one-out comparison datasets')
    parser.add_argument('--individual-benchmarks', type=str, choices=['true', 'false'], default='true', help='Generate individual benchmark datasets (true/false)')
    parser.add_argument('--all-benchmarks', type=str, choices=['true', 'false'], default='true', help='Generate all_benchmarks.csv aggregated dataset (true/false)')
    parser.add_argument('--sub-benchmarks', nargs='+', default=None, help='List of sub-benchmarks to generate')
    args = parser.parse_args()
    iterations = args.iterations
    data_type = args.data_type
    evaluation_type = args.evaluation_type
    compare = args.compare
    individual_benchmarks = args.individual_benchmarks.lower() == 'true'
    all_benchmarks = args.all_benchmarks.lower() == 'true'
    sub_benchmarks = args.sub_benchmarks
    
    # Use specified sub-benchmarks or default to all
    if sub_benchmarks:
        # Filter BENCHMARKS to only include specified sub-benchmarks
        filtered_benchmarks = []
        for benchmark_name, fid in BENCHMARKS:
            if benchmark_name in sub_benchmarks:
                filtered_benchmarks.append((benchmark_name, fid))
        benchmarks_to_generate = filtered_benchmarks
    else:
        benchmarks_to_generate = BENCHMARKS
    if args.output_root is not None:
        output_root = os.path.abspath(args.output_root)
    else:
        output_root = os.path.join(OUTPUT_ROOT, str(iterations))
    
    # Generate individual benchmark data (either permanent or temporary)
    benchmark_csvs = []
    temp_dirs = []
    
    if individual_benchmarks:
        # Generate individual benchmarks in the main output directory
        for benchmark_name, fid in benchmarks_to_generate:
            output_dir = os.path.join(output_root, benchmark_name)
            csv_path = run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir, data_type, evaluation_type)
            benchmark_csvs.append((benchmark_name, csv_path))
        print(f"Generated individual benchmark datasets: {[name for name, _ in benchmark_csvs]}")
    else:
        # Generate individual benchmarks in a temporary directory
        import tempfile
        temp_root = tempfile.mkdtemp(prefix="psa_temp_")
        print(f"Generating temporary individual benchmark datasets in: {temp_root}")
        
        for benchmark_name, fid in benchmarks_to_generate:
            output_dir = os.path.join(temp_root, benchmark_name)
            csv_path = run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir, data_type, evaluation_type)
            benchmark_csvs.append((benchmark_name, csv_path))
            temp_dirs.append(output_dir)
        print(f"Generated temporary individual benchmark datasets: {[name for name, _ in benchmark_csvs]}")
    
    # Generate all_benchmarks.csv if enabled
    if all_benchmarks and benchmark_csvs:
        aggregate_csvs(benchmark_csvs, os.path.join(output_root, 'all_benchmarks.csv'), data_type, evaluation_type)
        print(f"Aggregated CSV written to {os.path.join(output_root, 'all_benchmarks.csv')}")
    
    # Create leave-one-out datasets if compare is enabled
    if compare and benchmark_csvs:
        create_leave_one_out_datasets(benchmark_csvs, output_root, data_type, evaluation_type)
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