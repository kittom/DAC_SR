import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def load_cmaes_data(data_type='continuous'):
    """
    Load CMA-ES ground truth data.
    """
    data_path = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/CMAES/{data_type}/GTCMAES.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    # Load data with column names
    columns = ['FunctionID', 'Dimension', 'Repetition', 'Algorithm', 'Lambda', 'PtNormLog', 'ScaleFactor', 'Precision', 'UsedBudget']
    df = pd.read_csv(data_path, header=None, names=columns)
    
    return df

def load_optimal_policy_data():
    """
    Load optimal policy data.
    """
    data_path = os.path.join(PROJECT_ROOT, 'DataSets/Ground_Truth/CMAES/continuous/GTCMAES_OptimalPolicy.csv')
    
    if not os.path.exists(data_path):
        print(f"Optimal policy file not found: {data_path}")
        return None
    
    # Load data with column names
    columns = ['FunctionID', 'Dimension', 'Lambda', 'PtNormLog', 'ScaleFactor', 'OptimalLambda']
    df = pd.read_csv(data_path, header=None, names=columns)
    
    return df

def plot_lambda_distribution(df, save_dir):
    """
    Plot distribution of population sizes across different algorithms.
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different algorithms
    algorithms = df['Algorithm'].unique()
    n_algorithms = len(algorithms)
    
    for i, algorithm in enumerate(algorithms):
        plt.subplot(2, 3, i + 1)
        algo_data = df[df['Algorithm'] == algorithm]['Lambda']
        
        plt.hist(algo_data, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{algorithm.upper()} Population Size Distribution')
        plt.xlabel('Population Size (λ)')
        plt.ylabel('Frequency')
        plt.axvline(algo_data.mean(), color='red', linestyle='--', label=f'Mean: {algo_data.mean():.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lambda_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_space_analysis(df, save_dir):
    """
    Plot analysis of the state space components.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Lambda vs PtNormLog
    scatter1 = axes[0, 0].scatter(df['PtNormLog'], df['Lambda'], c=df['Precision'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_xlabel('PtNorm (Log Scale)')
    axes[0, 0].set_ylabel('Population Size (λ)')
    axes[0, 0].set_title('Population Size vs PtNorm')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Precision')
    
    # Lambda vs ScaleFactor
    scatter2 = axes[0, 1].scatter(df['ScaleFactor'], df['Lambda'], c=df['Precision'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('Scale Factor')
    axes[0, 1].set_ylabel('Population Size (λ)')
    axes[0, 1].set_title('Population Size vs Scale Factor')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Precision')
    
    # PtNormLog vs ScaleFactor
    scatter3 = axes[1, 0].scatter(df['PtNormLog'], df['ScaleFactor'], c=df['Lambda'], cmap='plasma', alpha=0.6)
    axes[1, 0].set_xlabel('PtNorm (Log Scale)')
    axes[1, 0].set_ylabel('Scale Factor')
    axes[1, 0].set_title('State Space: PtNorm vs Scale Factor')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Population Size (λ)')
    
    # Precision over time (UsedBudget)
    for algorithm in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algorithm]
        mean_precision = algo_data.groupby('UsedBudget')['Precision'].mean()
        axes[1, 1].plot(mean_precision.index, mean_precision.values, label=algorithm.upper(), linewidth=2)
    
    axes[1, 1].set_xlabel('Used Budget')
    axes[1, 1].set_ylabel('Mean Precision')
    axes[1, 1].set_title('Precision Evolution Over Time')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'state_space_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_algorithm_comparison(df, save_dir):
    """
    Compare performance across different algorithms.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Final precision comparison
    final_data = df.groupby(['Algorithm', 'FunctionID', 'Dimension', 'Repetition'])['Precision'].last().reset_index()
    sns.boxplot(data=final_data, x='Algorithm', y='Precision', ax=axes[0, 0])
    axes[0, 0].set_title('Final Precision by Algorithm')
    axes[0, 0].set_yscale('log')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average lambda by algorithm
    avg_lambda = df.groupby('Algorithm')['Lambda'].mean().reset_index()
    axes[0, 1].bar(avg_lambda['Algorithm'], avg_lambda['Lambda'])
    axes[0, 1].set_title('Average Population Size by Algorithm')
    axes[0, 1].set_ylabel('Average λ')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision convergence
    for algorithm in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algorithm]
        convergence = algo_data.groupby('UsedBudget')['Precision'].mean()
        axes[1, 0].plot(convergence.index, convergence.values, label=algorithm.upper(), linewidth=2)
    
    axes[1, 0].set_xlabel('Used Budget')
    axes[1, 0].set_ylabel('Mean Precision')
    axes[1, 0].set_title('Convergence Comparison')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    
    # Lambda evolution over time
    for algorithm in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algorithm]
        lambda_evolution = algo_data.groupby('UsedBudget')['Lambda'].mean()
        axes[1, 1].plot(lambda_evolution.index, lambda_evolution.values, label=algorithm.upper(), linewidth=2)
    
    axes[1, 1].set_xlabel('Used Budget')
    axes[1, 1].set_ylabel('Mean Population Size (λ)')
    axes[1, 1].set_title('Population Size Evolution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_function_analysis(df, save_dir):
    """
    Analyze performance across different BBOB functions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Final precision by function
    final_data = df.groupby(['FunctionID', 'Algorithm'])['Precision'].last().reset_index()
    sns.boxplot(data=final_data, x='FunctionID', y='Precision', hue='Algorithm', ax=axes[0, 0])
    axes[0, 0].set_title('Final Precision by Function ID')
    axes[0, 0].set_yscale('log')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average lambda by function
    avg_lambda_func = df.groupby(['FunctionID', 'Algorithm'])['Lambda'].mean().reset_index()
    sns.boxplot(data=avg_lambda_func, x='FunctionID', y='Lambda', hue='Algorithm', ax=axes[0, 1])
    axes[0, 1].set_title('Average Population Size by Function ID')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Dimension analysis
    final_data_dim = df.groupby(['Dimension', 'Algorithm'])['Precision'].last().reset_index()
    sns.boxplot(data=final_data_dim, x='Dimension', y='Precision', hue='Algorithm', ax=axes[1, 0])
    axes[1, 0].set_title('Final Precision by Dimension')
    axes[1, 0].set_yscale('log')
    
    # State space coverage
    scatter = axes[1, 1].scatter(df['PtNormLog'], df['ScaleFactor'], c=df['FunctionID'], cmap='tab20', alpha=0.6)
    axes[1, 1].set_xlabel('PtNorm (Log Scale)')
    axes[1, 1].set_ylabel('Scale Factor')
    axes[1, 1].set_title('State Space Coverage by Function')
    plt.colorbar(scatter, ax=axes[1, 1], label='Function ID')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'function_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimal_policy_analysis(optimal_df, save_dir):
    """
    Analyze the optimal policy data.
    """
    if optimal_df is None:
        print("No optimal policy data available for analysis.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Optimal lambda distribution
    axes[0, 0].hist(optimal_df['OptimalLambda'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Optimal Population Size (λ)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Optimal Population Sizes')
    axes[0, 0].axvline(optimal_df['OptimalLambda'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {optimal_df["OptimalLambda"].mean():.1f}')
    axes[0, 0].legend()
    
    # Optimal lambda vs state components
    scatter1 = axes[0, 1].scatter(optimal_df['PtNormLog'], optimal_df['OptimalLambda'], 
                                 c=optimal_df['ScaleFactor'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('PtNorm (Log Scale)')
    axes[0, 1].set_ylabel('Optimal Population Size (λ)')
    axes[0, 1].set_title('Optimal Policy: λ vs PtNorm')
    plt.colorbar(scatter1, ax=axes[0, 1], label='Scale Factor')
    
    # Optimal lambda by function
    sns.boxplot(data=optimal_df, x='FunctionID', y='OptimalLambda', ax=axes[1, 0])
    axes[1, 0].set_title('Optimal Population Size by Function ID')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Optimal lambda by dimension
    sns.boxplot(data=optimal_df, x='Dimension', y='OptimalLambda', ax=axes[1, 1])
    axes[1, 1].set_title('Optimal Population Size by Dimension')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimal_policy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics(df, optimal_df, save_dir):
    """
    Create summary statistics and save to file.
    """
    summary_stats = []
    
    # Overall statistics
    summary_stats.append("=== CMA-ES Dataset Summary ===")
    summary_stats.append(f"Total data points: {len(df)}")
    summary_stats.append(f"Functions tested: {df['FunctionID'].nunique()}")
    summary_stats.append(f"Dimensions tested: {df['Dimension'].nunique()}")
    summary_stats.append(f"Algorithms tested: {df['Algorithm'].nunique()}")
    summary_stats.append(f"Repetitions per configuration: {df['Repetition'].nunique()}")
    
    # Algorithm performance
    summary_stats.append("\n=== Algorithm Performance ===")
    final_precision = df.groupby(['Algorithm', 'FunctionID', 'Dimension', 'Repetition'])['Precision'].last()
    for algorithm in df['Algorithm'].unique():
        algo_precision = final_precision.xs(algorithm, level=0)
        summary_stats.append(f"{algorithm.upper()}:")
        summary_stats.append(f"  Mean final precision: {algo_precision.mean():.2e}")
        summary_stats.append(f"  Std final precision: {algo_precision.std():.2e}")
        summary_stats.append(f"  Mean population size: {df[df['Algorithm'] == algorithm]['Lambda'].mean():.1f}")
    
    # State space statistics
    summary_stats.append("\n=== State Space Statistics ===")
    summary_stats.append(f"Lambda range: [{df['Lambda'].min():.1f}, {df['Lambda'].max():.1f}]")
    summary_stats.append(f"PtNorm (log) range: [{df['PtNormLog'].min():.3f}, {df['PtNormLog'].max():.3f}]")
    summary_stats.append(f"Scale factor range: [{df['ScaleFactor'].min():.3f}, {df['ScaleFactor'].max():.3f}]")
    
    # Optimal policy statistics
    if optimal_df is not None:
        summary_stats.append("\n=== Optimal Policy Statistics ===")
        summary_stats.append(f"Optimal policy data points: {len(optimal_df)}")
        summary_stats.append(f"Optimal lambda range: [{optimal_df['OptimalLambda'].min():.1f}, {optimal_df['OptimalLambda'].max():.1f}]")
        summary_stats.append(f"Mean optimal lambda: {optimal_df['OptimalLambda'].mean():.1f}")
    
    # Save summary
    with open(os.path.join(save_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('\n'.join(summary_stats))
    
    print("Summary statistics saved to summary_statistics.txt")

def main():
    """
    Main function to generate all visualizations.
    """
    # Create output directory
    save_dir = os.path.join(PROJECT_ROOT, 'DataSets/Ground_Truth/CMAES/continuous/Visualisations')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading CMA-ES data...")
    df = load_cmaes_data('continuous')
    
    if df is None:
        print("No data found. Please run the CMA-ES generator first.")
        return
    
    print(f"Loaded {len(df)} data points")
    
    # Load optimal policy data
    optimal_df = load_optimal_policy_data()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    plot_lambda_distribution(df, save_dir)
    print("✓ Lambda distribution plot saved")
    
    plot_state_space_analysis(df, save_dir)
    print("✓ State space analysis plot saved")
    
    plot_algorithm_comparison(df, save_dir)
    print("✓ Algorithm comparison plot saved")
    
    plot_function_analysis(df, save_dir)
    print("✓ Function analysis plot saved")
    
    if optimal_df is not None:
        plot_optimal_policy_analysis(optimal_df, save_dir)
        print("✓ Optimal policy analysis plot saved")
    
    create_summary_statistics(df, optimal_df, save_dir)
    print("✓ Summary statistics saved")
    
    print(f"\nAll visualizations saved to: {save_dir}")

if __name__ == "__main__":
    main() 