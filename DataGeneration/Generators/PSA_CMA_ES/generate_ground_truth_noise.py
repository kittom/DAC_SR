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

def run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir, data_type='continuous', noise_level=0.01, noise_type='gaussian'):
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
            # Add noise
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_level, noise_level)
            else:
                raise ValueError("noise_type must be 'gaussian' or 'uniform'")
            next_lambda_unrounded += noise
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
                gamma_theta = (1 - psa_beta)**2 * gamma_theta + psa_beta * (2 - psa_beta)
                next_lambda_unrounded = lambda_ * np.exp(psa_beta * (gamma_theta - (ptnorm / alpha)))
                # Add noise
                if noise_type == 'gaussian':
                    noise = np.random.normal(0, noise_level)
                elif noise_type == 'uniform':
                    noise = np.random.uniform(-noise_level, noise_level)
                else:
                    raise ValueError("noise_type must be 'gaussian' or 'uniform'")
                next_lambda_unrounded += noise
                if data_type == 'discrete':
                    next_lambda_unrounded = round(next_lambda_unrounded)
                writer.writerow([
                    lambda_, psa_beta, ptnorm, alpha, gamma_theta, next_lambda_unrounded
                ])
        # Write results files based on data type with ground truth as first column
        if data_type == 'continuous':
            # Create two results files for continuous data
            # 1. For control library evaluation (results.csv)
            control_results_path = os.path.join(output_dir, 'results.csv')
            with open(control_results_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ground_truth'])
                writer.writerow([ground_truth_equation])
            # 2. For tailored library evaluation (results_lib.csv)
            tailored_results_path = os.path.join(output_dir, 'results_lib.csv')
            with open(tailored_results_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ground_truth'])
                writer.writerow([ground_truth_equation])
        else:  # discrete
            # Create one results file for discrete data (rounding evaluation)
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

def aggregate_csvs(benchmark_csvs, output_path, data_type='continuous'):
    rows = []
    for benchmark_name, csv_path in benchmark_csvs:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    ground_truth_equation = 'x1 * exp(x2 * (x5 - (x3 / x4)))'
    if data_type == 'continuous':
        control_results_path = os.path.join(os.path.dirname(output_path), 'results.csv')
        with open(control_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
        tailored_results_path = os.path.join(os.path.dirname(output_path), 'results_lib.csv')
        with open(tailored_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])
    else:
        rounding_results_path = os.path.join(os.path.dirname(output_path), 'results_rounding.csv')
        with open(rounding_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth'])
            writer.writerow([ground_truth_equation])

def main():
    parser = argparse.ArgumentParser(description="Generate PSA-CMA-ES ground truth data for multiple benchmarks with noise.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of generations to run for each benchmark')
    parser.add_argument('--output-root', type=str, default=None, help='Optional output root directory for CSVs')
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous', help='Type of data to generate')
    parser.add_argument('--noise-level', type=float, default=0.01, help='Noise level (std for gaussian, range for uniform)')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian', help='Type of noise to add')
    args = parser.parse_args()
    iterations = args.iterations
    data_type = args.data_type
    noise_level = args.noise_level
    noise_type = args.noise_type
    if args.output_root is not None:
        output_root = os.path.abspath(args.output_root)
    else:
        output_root = os.path.join(OUTPUT_ROOT, str(iterations))
    benchmark_csvs = []
    for benchmark_name, fid in BENCHMARKS:
        output_dir = os.path.join(output_root, benchmark_name)
        csv_path = run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir, data_type, noise_level, noise_type)
        benchmark_csvs.append((benchmark_name, csv_path))
    aggregate_csvs(benchmark_csvs, os.path.join(output_root, 'all_benchmarks.csv'), data_type)
    print(f"Aggregated CSV written to {os.path.join(output_root, 'all_benchmarks.csv')}")

if __name__ == "__main__":
    main() 