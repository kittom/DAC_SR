# CMA-ES Benchmark Summary

## Overview

This document summarizes the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) benchmark that has been added to the DataSets directory. This benchmark complements the existing OneMax and LeadingOnes benchmarks by providing a more complex, continuous optimization scenario.

## What is CMA-ES?

CMA-ES is a state-of-the-art derivative-free optimization algorithm that adapts its search distribution based on the optimization landscape. The key innovation in this benchmark is **Population Size Adaptation (PSA)**, where we learn to control the population size parameter (λ) dynamically during optimization.

## State Space

The policy receives a 3-dimensional continuous state:

1. **Lambda (λ)**: Current population size [4, 512]
2. **PtNorm (Log Scale)**: Evolution path norm on logarithmic scale [0, 5]
3. **Scale Factor**: Expected update step size norm [0.1, 10]

## Action Space

The policy outputs a continuous population size value that is clamped between 4 and 512.

## Reward Function

The reward is the negative logarithm of the current precision:
```
reward = -log(current_precision)
```
where `current_precision = best_found_value - global_optimum`

## Dataset Structure

### Raw Data (`GTCMAES.csv`)
- **FunctionID**: BBOB function ID (1-24)
- **Dimension**: Problem dimension (10, 20, 30)
- **Repetition**: Repetition number
- **Algorithm**: Population size adaptation algorithm
- **Lambda**: Current population size
- **PtNormLog**: Evolution path norm (log scale)
- **ScaleFactor**: Expected update step size norm
- **Precision**: Current precision
- **UsedBudget**: Function evaluations used

### Optimal Policy Data (`GTCMAES_OptimalPolicy.csv`)
- **FunctionID**: BBOB function ID
- **Dimension**: Problem dimension
- **Lambda**: State lambda bin center
- **PtNormLog**: State PtNorm bin center
- **ScaleFactor**: State scale factor bin center
- **OptimalLambda**: Optimal population size for this state

## Algorithms Tested

1. **PSA**: Population Size Adaptation (target policy)
2. **lin-inc**: Linear increase
3. **lin-dec**: Linear decrease
4. **exp-inc**: Exponential increase
5. **exp-dec**: Exponential decrease

## Key Differences from Existing Benchmarks

| Aspect | OneMax/LeadingOnes | CMA-ES |
|--------|-------------------|---------|
| **State Space** | Discrete, 1D | Continuous, 3D |
| **Action Space** | Discrete | Continuous |
| **Problem Type** | Theoretical | Real optimization |
| **Complexity** | Simple | Complex |
| **State Meaning** | Current fitness | Optimization state |
| **Action Meaning** | Bitflip probability | Population size |

## Files Created

### Generators
- `CMAESGT.py`: Main data generator
- `visualise_data.py`: Visualization script
- `test_data_usage.py`: Usage demonstration
- `requirements.txt`: Dependencies
- `README.md`: Detailed documentation

### Generated Data
- `DataSets/Ground_Truth/CMAES/continuous/GTCMAES.csv`: Raw data
- `DataSets/Ground_Truth/CMAES/continuous/GTCMAES_OptimalPolicy.csv`: Optimal policy
- `DataSets/Ground_Truth/CMAES/continuous/Visualisations/`: Analysis plots

## Usage Examples

### Basic Data Generation
```bash
# Generate with default settings
python CMAESGT.py

# Generate for specific functions and dimensions
python CMAESGT.py 1 2 3 10 20 30

# Generate discrete population sizes
python CMAESGT.py --data-type discrete 1 2 10
```

### Data Loading
```python
import pandas as pd

# Load raw data
columns = ['FunctionID', 'Dimension', 'Repetition', 'Algorithm', 'Lambda', 'PtNormLog', 'ScaleFactor', 'Precision', 'UsedBudget']
df = pd.read_csv('GTCMAES.csv', header=None, names=columns)

# Load optimal policy
optimal_columns = ['FunctionID', 'Dimension', 'Lambda', 'PtNormLog', 'ScaleFactor', 'OptimalLambda']
optimal_df = pd.read_csv('GTCMAES_OptimalPolicy.csv', header=None, names=optimal_columns)
```

### Training Setup
```python
# Extract state-action pairs for training
X = optimal_df[['Lambda', 'PtNormLog', 'ScaleFactor']].values  # States
y = optimal_df['OptimalLambda'].values  # Optimal actions

# Split by function/dimension for cross-validation
for fid in optimal_df['FunctionID'].unique():
    for dim in optimal_df['Dimension'].unique():
        config_data = optimal_df[(optimal_df['FunctionID'] == fid) & (optimal_df['Dimension'] == dim)]
        # Use config_data for training/validation
```

## Integration with Existing Codebase

The CMA-ES benchmark integrates seamlessly with the existing PSA-CMA-ES codebase:

1. **State Representation**: Matches the environment state in `modcma_pop_size_R1.py`
2. **Action Space**: Compatible with the environment's action space [4, 512]
3. **Reward Function**: Uses the same reward calculation as the environment
4. **Instance Sets**: Uses the same BBOB functions and dimensions

## Experimental Design Considerations

### Complexity Factors
1. **Continuous State Space**: Unlike discrete benchmarks, requires function approximation
2. **Multiple Dimensions**: 3D state space vs 1D in existing benchmarks
3. **Real Optimization**: Uses actual optimization problems vs theoretical fitness functions
4. **Dynamic Adaptation**: Population size changes during optimization

### Dataset Size Considerations
- **Raw Data**: 4500+ data points per configuration
- **Optimal Policy**: 3888+ state-action pairs
- **Multiple Functions**: 24 BBOB functions available
- **Multiple Dimensions**: Scalable to different problem sizes

### Validation Strategy
- **Function-wise**: Test on unseen BBOB functions
- **Dimension-wise**: Test on unseen problem dimensions
- **Algorithm-wise**: Compare against baseline adaptation strategies

## Future Extensions

1. **More Functions**: Extend to all 24 BBOB functions
2. **Higher Dimensions**: Test on larger problem dimensions
3. **Different Budgets**: Vary optimization budget
4. **Noise**: Add noise to state observations
5. **Transfer Learning**: Study generalization across functions

## Comparison with Existing Benchmarks

### OneMax Benchmark
- **Purpose**: Learn optimal bitflip probability
- **State**: Current fitness level
- **Action**: Bitflip probability
- **Complexity**: Low

### LeadingOnes Benchmark
- **Purpose**: Learn optimal bitflip probability
- **State**: Current fitness level
- **Action**: Bitflip probability
- **Complexity**: Low

### CMA-ES Benchmark
- **Purpose**: Learn optimal population size adaptation
- **State**: 3D optimization state
- **Action**: Population size
- **Complexity**: High

## Conclusion

The CMA-ES benchmark provides a significant step up in complexity from the existing benchmarks while maintaining the same structure and format. It enables research into:

1. **Continuous control** in reinforcement learning
2. **Multi-dimensional state spaces**
3. **Real optimization problems**
4. **Dynamic parameter adaptation**

This benchmark is particularly valuable for studying how reinforcement learning can be applied to real-world optimization scenarios where the optimal policy depends on the current state of the optimization process. 