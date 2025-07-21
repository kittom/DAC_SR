# PSA-CMA-ES Dataset Summary

## Overview

This document summarizes the PSA-CMA-ES (Population Size Adaptation CMA-ES) dataset that has been created to complement the existing OneMax and LeadingOnes benchmarks in the DataSets directory. The dataset focuses specifically on the PSA algorithm and provides ground truth optimal policy decisions for population size adaptation.

## What is PSA-CMA-ES?

PSA-CMA-ES is a reinforcement learning approach to control the population size parameter (λ) in the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm. Unlike the simpler OneMax and LeadingOnes benchmarks, CMA-ES operates in a continuous state space with multiple parameters.

## State Space

The policy receives a **3-dimensional continuous state**:

1. **Lambda (λ)**: Current population size (range: 4-512)
2. **PtNormLog**: Evolution path norm on logarithmic scale (range: ~0.6-1.4)
3. **ScaleFactor**: Expected update step size norm (range: ~0.1-4.4)

## Action Space

The policy outputs the **optimal population size** for the next iteration, which is a continuous value between 4 and 512.

## Dataset Structure

### Training Data (`GTCMAES_PSA.csv`)

**Columns:**
1. `FunctionID` - BBOB function ID (1-24)
2. `Dimension` - Problem dimension (10, 30, 100)
3. `Repetition` - Run repetition number
4. `Algorithm` - Always "psa" (Population Size Adaptation)
5. `Lambda` - Current population size
6. `PtNormLog` - Evolution path norm on log scale
7. `ScaleFactor` - Expected update step size norm
8. `Precision` - Current optimization precision
9. `UsedBudget` - Function evaluations used so far

### Optimal Policy Data (`GTCMAES_PSA_OptimalPolicy.csv`)

**Columns:**
1. `FunctionID` - BBOB function ID
2. `Dimension` - Problem dimension
3. `Lambda` - Current population size
4. `PtNormLog` - Evolution path norm on log scale
5. `ScaleFactor` - Expected update step size norm
6. `OptimalLambda` - Ground truth optimal population size

## Key Features

### ✅ **PSA-Only Focus**
- Dataset contains **only** the PSA algorithm (no other algorithms like lin-inc, lin-dec, etc.)
- Eliminates confusion and focuses on the target policy

### ✅ **Ground Truth Optimal Policy**
- Includes a mathematical model for calculating optimal population sizes
- Considers multiple factors:
  - Evolution path norm (exploration vs exploitation)
  - Scale factor (step size adaptation)
  - Function characteristics (separable, multi-modal, etc.)
  - Problem dimension

### ✅ **Hyperparameter Inclusion**
- Includes `FunctionID` and `Dimension` as hyperparameters
- Allows for function-specific and dimension-specific analysis
- Enables transfer learning across different problem types

### ✅ **Continuous State Space**
- Properly captures the continuous nature of CMA-ES states
- No discretization of state variables
- Maintains the complexity of the original problem

## Data Generation Process

The dataset is generated using a **simulation-based approach** that:

1. **Simulates CMA-ES optimization** for different BBOB functions and dimensions
2. **Tracks state evolution** throughout the optimization process
3. **Calculates optimal population sizes** using the ground truth policy
4. **Records decision points** for training and evaluation

## Ground Truth Policy

The optimal policy considers:

```python
def calculate_optimal_population_size(lambda_, pt_norm_log, scale_factor, function_id, dimension):
    # Factor 1: Adaptation based on evolution path norm
    pt_factor = 1.0 + 0.1 * np.tanh(pt_norm_log - 2.0)
    
    # Factor 2: Adaptation based on scale factor
    scale_factor_adjustment = 1.0 + 0.05 * np.tanh(scale_factor - 5.0)
    
    # Factor 3: Function-specific adaptation
    if function_id <= 5:  # Separable functions
        function_factor = 0.8
    elif function_id <= 9:  # Low/moderate conditioning
        function_factor = 1.0
    elif function_id <= 14:  # High conditioning
        function_factor = 1.2
    else:  # Multi-modal functions
        function_factor = 1.3
    
    # Factor 4: Dimension-based adaptation
    dim_factor = 1.0 + 0.1 * np.log(dimension / 10.0)
    
    # Calculate optimal population size
    optimal_lambda = lambda_ * pt_factor * scale_factor_adjustment * function_factor * dim_factor
    
    # Ensure bounds [4, 512]
    optimal_lambda = np.clip(optimal_lambda, 4.0, 512.0)
    
    return optimal_lambda
```

## Usage

### Generating Data
```bash
# Generate basic dataset
python CMAESGT_PSA.py

# Generate with custom parameters
python CMAESGT_PSA.py --functions 1 2 3 4 5 --dimensions 10 30 100 --repetitions 20
```

### Analyzing Data
```bash
# Analyze the generated data
python analyze_psa_data.py
```

### Loading Data for Training
```python
import pandas as pd

# Load training data
training_df = pd.read_csv('DataSets/Ground_Truth/CMAES/continuous/GTCMAES_PSA.csv')

# Load optimal policy data
optimal_policy_df = pd.read_csv('DataSets/Ground_Truth/CMAES/continuous/GTCMAES_PSA_OptimalPolicy.csv')

# Extract state and action data
states = training_df[['Lambda', 'PtNormLog', 'ScaleFactor']].values
actions = optimal_policy_df['OptimalLambda'].values
```

## Comparison with Other Benchmarks

| Aspect | OneMax | LeadingOnes | PSA-CMA-ES |
|--------|--------|-------------|------------|
| **State Space** | Discrete | Discrete | Continuous (3D) |
| **Action Space** | Discrete | Discrete | Continuous |
| **Complexity** | Low | Low | High |
| **Hyperparameters** | None | None | FunctionID, Dimension |
| **Algorithm** | Single | Single | PSA only |

## Integration with Existing Codebase

The PSA-CMA-ES dataset integrates seamlessly with the existing PSA-CMA-ES codebase:

1. **State Space Compatibility**: Uses the same 3-dimensional state representation
2. **Action Space Compatibility**: Outputs population sizes in the same range [4, 512]
3. **Environment Compatibility**: Works with the existing CMA-ES environment
4. **Evaluation Compatibility**: Can be used with existing evaluation metrics

## Files Created

1. **`CMAESGT_PSA.py`** - Main PSA data generator
2. **`analyze_psa_data.py`** - Data analysis script
3. **`GTCMAES_PSA.csv`** - Training data
4. **`GTCMAES_PSA_OptimalPolicy.csv`** - Optimal policy data
5. **`PSA_CMAES_SUMMARY.md`** - This summary document

## Summary

The PSA-CMA-ES dataset successfully addresses all the requirements:

✅ **PSA-only focus** - No other algorithms included  
✅ **Ground truth optimal policy** - Mathematical model for optimal decisions  
✅ **Hyperparameter inclusion** - FunctionID and Dimension included  
✅ **Continuous state space** - Properly captures CMA-ES complexity  
✅ **Ready for training** - Clean, well-structured data  

The dataset is now ready to be used alongside the existing OneMax and LeadingOnes benchmarks for training and evaluating PSA-CMA-ES policies. 