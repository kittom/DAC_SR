#!/usr/bin/env python3
"""
Script to compare results between Experiment 1 (full library) and Experiment 2 (minimal library)
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_results(results_file):
    """Load results from a CSV file."""
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return None

def compare_datasets(exp1_dir, output_file="experiment_comparison.csv"):
    """Compare results between full library and minimal library within Experiment 1."""
    
    comparison_data = []
    
    # Get dataset directories
    exp1_datasets = Path(exp1_dir) / "Datasets"
    
    if not exp1_datasets.exists():
        print("Error: Dataset directory not found!")
        return
    
    # Compare OneMax datasets
    onemax_dir = exp1_datasets / "OneMax"
    if onemax_dir.exists():
        csv_file = onemax_dir / "GTOneMax.csv"
        if csv_file.exists():
            dataset_name = csv_file.stem
            full_results = load_results(onemax_dir / "results.csv")
            minimal_results = load_results(onemax_dir / "results_lib.csv")
            
            if full_results is not None and minimal_results is not None:
                comparison_data.append({
                    'dataset': dataset_name,
                    'problem_type': 'one_max',
                    'full_library_has_results': True,
                    'minimal_library_has_results': True,
                    'full_library_equations': len(full_results.columns) - 1,  # Exclude ground_truth
                    'minimal_library_equations': len(minimal_results.columns) - 1
                })
            else:
                comparison_data.append({
                    'dataset': dataset_name,
                    'problem_type': 'one_max',
                    'full_library_has_results': full_results is not None,
                    'minimal_library_has_results': minimal_results is not None,
                    'full_library_equations': 0,
                    'minimal_library_equations': 0
                })
    
    # Compare LeadingOnes datasets
    leadingones_dir = exp1_datasets / "LeadingOnes"
    if leadingones_dir.exists():
        csv_file = leadingones_dir / "GTLeadingOnes.csv"
        if csv_file.exists():
            dataset_name = csv_file.stem
            full_results = load_results(leadingones_dir / "results.csv")
            minimal_results = load_results(leadingones_dir / "results_lib.csv")
            
            if full_results is not None and minimal_results is not None:
                comparison_data.append({
                    'dataset': dataset_name,
                    'problem_type': 'leading_ones',
                    'full_library_has_results': True,
                    'minimal_library_has_results': True,
                    'full_library_equations': len(full_results.columns) - 1,
                    'minimal_library_equations': len(minimal_results.columns) - 1
                })
            else:
                comparison_data.append({
                    'dataset': dataset_name,
                    'problem_type': 'leading_ones',
                    'full_library_has_results': full_results is not None,
                    'minimal_library_has_results': minimal_results is not None,
                    'full_library_equations': 0,
                    'minimal_library_equations': 0
                })
    
    # Compare PSA-CMA-ES datasets
    psacmaes_dir = exp1_datasets / "PSACMAES"
    if psacmaes_dir.exists():
        for benchmark_dir in psacmaes_dir.iterdir():
            if benchmark_dir.is_dir():
                benchmark_name = benchmark_dir.name
                psa_vars_file = benchmark_dir / "psa_vars.csv"
                
                if psa_vars_file.exists():
                    full_results = load_results(benchmark_dir / "results.csv")
                    minimal_results = load_results(benchmark_dir / "results_lib.csv")
                    
                    if full_results is not None and minimal_results is not None:
                        comparison_data.append({
                            'dataset': f"psacmaes_{benchmark_name}",
                            'problem_type': 'psa',
                            'full_library_has_results': True,
                            'minimal_library_has_results': True,
                            'full_library_equations': len(full_results.columns) - 1,
                            'minimal_library_equations': len(minimal_results.columns) - 1
                        })
                    else:
                        comparison_data.append({
                            'dataset': f"psacmaes_{benchmark_name}",
                            'problem_type': 'psa',
                            'full_library_has_results': full_results is not None,
                            'minimal_library_has_results': minimal_results is not None,
                            'full_library_equations': 0,
                            'minimal_library_equations': 0
                        })
    
    # Create comparison DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_file, index=False)
        
        print(f"Comparison saved to: {output_file}")
        print(f"Total datasets compared: {len(comparison_df)}")
        print(f"Full library datasets with results: {comparison_df['full_library_has_results'].sum()}")
        print(f"Minimal library datasets with results: {comparison_df['minimal_library_has_results'].sum()}")
        
        # Summary by problem type
        print("\nSummary by problem type:")
        for problem_type in comparison_df['problem_type'].unique():
            subset = comparison_df[comparison_df['problem_type'] == problem_type]
            print(f"  {problem_type}: {len(subset)} datasets")
            print(f"    Full library results: {subset['full_library_has_results'].sum()}")
            print(f"    Minimal library results: {subset['minimal_library_has_results'].sum()}")
    else:
        print("No datasets found for comparison!")

def main():
    parser = argparse.ArgumentParser(description="Compare full library vs minimal library results within Experiment 1")
    parser.add_argument("exp1_dir", help="Path to Experiment 1 directory")
    parser.add_argument("--output", default="experiment_comparison.csv", help="Output file for comparison")
    
    args = parser.parse_args()
    
    print("==========================================")
    print("Library Comparison Tool")
    print("==========================================")
    print(f"Experiment 1 Directory: {args.exp1_dir}")
    print(f"Comparing: results.csv vs results_lib.csv")
    print(f"Output file: {args.output}")
    print("==========================================")
    
    compare_datasets(args.exp1_dir, args.output)

if __name__ == "__main__":
    main() 