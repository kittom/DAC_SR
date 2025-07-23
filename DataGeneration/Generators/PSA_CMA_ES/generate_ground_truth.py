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
CSV_COLS = ['lambda', 'psa_beta', 'ptnorm', 'alpha', 'update_term', 'exp_update_value', 'next_lambda_unrounded']


def run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir):
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
            update_term = 1 - (ptnorm / alpha)
            exp_update_value = psa_beta * update_term
            next_lambda_unrounded = lambda_ * np.exp(exp_update_value)
            writer.writerow([
                lambda_, psa_beta, ptnorm, alpha, update_term, exp_update_value, next_lambda_unrounded
            ])
            for _ in range(iterations-1):
                cma.step()
                lambda_ = cma.parameters.lambda_
                psa_beta = cma.parameters.psa_beta
                ptnorm = cma.parameters.ptnorm
                alpha = cma.parameters.alpha
                update_term = 1 - (ptnorm / alpha)
                exp_update_value = psa_beta * update_term
                next_lambda_unrounded = lambda_ * np.exp(exp_update_value)
                writer.writerow([
                    lambda_, psa_beta, ptnorm, alpha, update_term, exp_update_value, next_lambda_unrounded
                ])
    except Exception as e:
        print(f"Error writing CSV for {benchmark_name}: {e}")
    print(f"{benchmark_name}: best y={problem.state.current_best.y}")
    problem.reset()
    return csv_path

def aggregate_csvs(benchmark_csvs, output_path):
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

def main():
    parser = argparse.ArgumentParser(description="Generate PSA-CMA-ES ground truth data for multiple benchmarks.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of generations to run for each benchmark')
    args = parser.parse_args()
    iterations = args.iterations
    output_root = os.path.join(OUTPUT_ROOT, str(iterations))
    benchmark_csvs = []
    for benchmark_name, fid in BENCHMARKS:
        output_dir = os.path.join(output_root, benchmark_name)
        csv_path = run_psa_cmaes_benchmark(benchmark_name, fid, iterations, output_dir)
        benchmark_csvs.append((benchmark_name, csv_path))
    aggregate_csvs(benchmark_csvs, os.path.join(output_root, 'all_benchmarks.csv'))
    print(f"Aggregated CSV written to {os.path.join(output_root, 'all_benchmarks.csv')}")

if __name__ == "__main__":
    main() 