# CMA-ES Ground Truth Generator

This generator creates datasets for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) population size adaptation benchmark. Unlike the simpler OneMax and LeadingOnes benchmarks, CMA-ES operates in a continuous state space with multiple parameters.

## Overview

The CMA-ES benchmark focuses on learning optimal population size adaptation policies for the CMA-ES optimization algorithm. The policy receives a 3-dimensional state:

1. **Lambda (λ)**: Current population size (4-512)
2. **PtNorm (Log Scale)**: Evolution path norm on logarithmic scale
3. **Scale Factor**: Expected update step size norm

The policy outputs the optimal population size for the next iteration.

## State Space

The CMA-ES environment provides a continuous 3-dimensional state space:

- **Lambda**: Population size ranging from 4 to 512
- **PtNorm (Log Scale)**: Evolution path norm, typically 0-5 on log scale
- **Scale Factor**: Expected update step size norm, typically 0.1-10

## Action Space

The policy outputs a continuous population size value that is clamped between 4 and 512.

## Reward Function

The reward is the negative logarithm of the current precision (difference between best found solution and global optimum).

## Dataset Structure

The generated dataset contains the following columns:

### Raw Data (`GTCMAES.csv`):
- `FunctionID`: BBOB function ID (1-24)
- `Dimension`: Problem dimension
- `Repetition`: Repetition number
- `Algorithm`: Population size adaptation algorithm used
- `Lambda`: Current population size
- `PtNormLog`: Evolution path norm (log scale)
- `ScaleFactor`: Expected update step size norm
- `Precision`: Current precision (best - optimum)
- `UsedBudget`: Function evaluations used so far

### Optimal Policy Data (`GTCMAES_OptimalPolicy.csv`):
- `FunctionID`: BBOB function ID
- `Dimension`: Problem dimension
- `Lambda`: State lambda bin center
- `PtNormLog`: State PtNorm bin center
- `ScaleFactor`: State scale factor bin center
- `OptimalLambda`: Optimal population size for this state

## Usage

### Basic Usage

```bash
# Generate data with default settings
python CMAESGT.py

# Generate data for specific functions and dimensions
python CMAESGT.py 1 2 3 10 20 30

# Generate discrete population sizes
python CMAESGT.py --data-type discrete 1 2 10 20

# Custom budget and repetitions
python CMAESGT.py --budget-factor 1000 --repetitions 3 1 2 10
```

### Command Line Arguments

- `--data-type`: 'continuous' or 'discrete' (default: 'continuous')
- `--budget-factor`: Budget multiplier (default: 2500)
- `--repetitions`: Number of repetitions per configuration (default: 5)
- Function IDs: BBOB function IDs (1-24)
- Dimensions: Problem dimensions

### Visualization

After generating data, create visualizations:

```bash
python visualise_data.py
```

This creates plots in `DataSets/Ground_Truth/CMAES/continuous/Visualisations/`:
- Lambda distribution across algorithms
- State space analysis
- Algorithm performance comparison
- Function-specific analysis
- Optimal policy analysis
- Summary statistics

## Algorithms Tested

The generator tests multiple population size adaptation strategies:

1. **PSA**: Population Size Adaptation (the target policy)
2. **lin-inc**: Linear increase
3. **lin-dec**: Linear decrease
4. **exp-inc**: Exponential increase
5. **exp-dec**: Exponential decrease

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations
- `ioh`: BBOB benchmark functions
- `scikit-learn`: Machine learning utilities (for optimal policy generation)

## Output Files

1. **`GTCMAES.csv`**: Raw CMA-ES execution data
2. **`GTCMAES_OptimalPolicy.csv`**: Optimal policy dataset
3. **Visualizations/**: Various analysis plots
4. **`summary_statistics.txt`**: Dataset summary

## Notes

- The generator requires the ModularCMAES library to be available in the thesis directory
- If ModularCMAES is not available, the generator falls back to simulation mode
- The optimal policy is determined by finding the population size that led to the best precision in each state bin
- The dataset is designed to be compatible with the existing OneMax and LeadingOnes benchmarks

## Integration with Existing Benchmarks

This CMA-ES benchmark complements the existing OneMax and LeadingOnes benchmarks by providing:

1. **Continuous state space** vs discrete
2. **Multiple state dimensions** vs single dimension
3. **Complex optimization landscape** vs simple fitness functions
4. **Real-world optimization scenario** vs theoretical problems

The generated datasets follow the same structure and format as the existing benchmarks for consistency. 