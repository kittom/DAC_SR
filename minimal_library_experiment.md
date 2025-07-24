# Minimal Library Symbolic Regression Experiment

## Overview

This experiment evaluates how symbolic regression algorithms perform when given only the minimal set of mathematical functions required to express the ground truth equation. This helps us understand:

1. **Overfitting Detection**: If algorithms perform better with minimal libraries, it suggests they were overfitting to the full library
2. **Algorithm Efficiency**: How well algorithms can find solutions when given only necessary functions
3. **Function Preference**: Which algorithms are more sensitive to function library choices

## Problem Types and Required Functions

### OneMax Problem
- **Ground Truth**: `sqrt(x1/(x1-x2))`
- **Required Functions**: `+`, `-`, `/`, `sqrt`
- **Minimal Library**: Only these 4 functions

### LeadingOnes Problem  
- **Ground Truth**: `x1/(x2 + 1)`
- **Required Functions**: `+`, `/`
- **Minimal Library**: Only these 2 functions

### PSA-CMA-ES Problem
- **Ground Truth**: `x1 * exp(x2 * (x5 - (x3 / x4)))`
- **Required Functions**: `*`, `-`, `/`, `exp`
- **Minimal Library**: Only these 4 functions

## Algorithm Configurations

### PySR
- **Configuration Files**: `SR_algorithms/PySR/configs/minimal_*.py`
- **Control**: `binary_operators` and `unary_operators` lists
- **Script**: `run_pysr_lib.py`

### KAN
- **Configuration Files**: `SR_algorithms/pykan/configs/minimal_*.py`
- **Control**: `lib` parameter in `auto_symbolic()`
- **Script**: `main_lib.py`

### DeepSR
- **Configuration Files**: `SR_algorithms/DeepSR/deep-symbolic-optimization/configs/minimal_*.json`
- **Control**: `function_set` in task configuration
- **Script**: `run_deepsr_lib.py`

### TPSR
- **Configuration Files**: `SR_algorithms/TPSR/configs/minimal_*.py`
- **Control**: `operators_real` dictionary with operator arity
- **Script**: `run_tpsr_lib.py`

### Linear Regression
- **Configuration**: No minimal library needed (always linear)
- **Script**: Uses original `run_linear_on_csv.py`

## Usage

### Running the Experiment

```bash
# Run minimal library experiment on OneMax dataset
./Scripts/run_all_library.sh one_max path/to/onemax.csv

# Run with custom noise threshold
./Scripts/run_all_library.sh leading_ones path/to/leadingones.csv 0.1

# Run on PSA-CMA-ES dataset
./Scripts/run_all_library.sh psa path/to/psacmaes.csv
```

### Individual Algorithm Scripts

```bash
# Run individual algorithms with minimal library
./Scripts/unrounded/algorithm_library/pysr_lib.sh path/to/data.csv one_max
./Scripts/unrounded/algorithm_library/kan_lib.sh path/to/data.csv leading_ones
./Scripts/unrounded/algorithm_library/deepsr_lib.sh path/to/data.csv psa
```

## Results

### Output Files
- **`results_lib.csv`**: Contains equations found by each algorithm with minimal library
- **`results.csv`**: Original results with full library (for comparison)

### Expected Outcomes

1. **Better Performance**: If minimal library performs better, suggests overfitting in full library
2. **Worse Performance**: If minimal library performs worse, suggests algorithms benefit from function diversity
3. **Similar Performance**: If performance is similar, suggests algorithms are robust to library choices

## Configuration Details

### PySR Minimal Configurations

```python
# OneMax: sqrt(x1/(x1-x2))
BINARY_OPERATORS = ["+", "-", "/"]
UNARY_OPERATORS = ["sqrt"]

# LeadingOnes: x1/(x2 + 1)  
BINARY_OPERATORS = ["+", "/"]
UNARY_OPERATORS = []

# PSA-CMA-ES: x1 * exp(x2 * (x5 - (x3 / x4)))
BINARY_OPERATORS = ["*", "-", "/"]
UNARY_OPERATORS = ["exp"]
```

### KAN Minimal Configurations

```python
# OneMax
MINIMAL_LIB = ['x', 'sqrt', '1/x']

# LeadingOnes
MINIMAL_LIB = ['x', '1/x']

# PSA-CMA-ES
MINIMAL_LIB = ['x', 'exp', '1/x']
```

### DeepSR Minimal Configurations

```json
// OneMax
"function_set": ["add", "sub", "div", "sqrt"]

// LeadingOnes
"function_set": ["add", "div"]

// PSA-CMA-ES
"function_set": ["mul", "sub", "div", "exp"]
```

### TPSR Minimal Configurations

```python
# OneMax
MINIMAL_OPERATORS = {
    "add": 2, "sub": 2, "div": 2, "sqrt": 1
}

# LeadingOnes
MINIMAL_OPERATORS = {
    "add": 2, "div": 2
}

# PSA-CMA-ES
MINIMAL_OPERATORS = {
    "mul": 2, "sub": 2, "div": 2, "exp": 1
}
```

## Comparison with Full Library

The full library includes all standard mathematical functions:
- **Binary**: `+`, `-`, `*`, `/`, `^`
- **Unary**: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `exp`, `log`, `sqrt`, `abs`, `tanh`
- **Special**: `x^2`, `x^3`, `1/x`

## Analysis

Compare `results.csv` (full library) with `results_lib.csv` (minimal library) to:

1. **Measure Overfitting**: Check if minimal library improves performance
2. **Function Sensitivity**: See which algorithms are most affected by library changes
3. **Convergence**: Compare iteration counts and convergence behavior
4. **Solution Quality**: Analyze equation complexity and accuracy

## Limitations

1. **Q-Lattice**: Cannot be easily configured for minimal libraries (automatic function discovery)
2. **E2E Transformer**: Pre-trained model, cannot modify function library
3. **TPSR**: Implementation may need adaptation for minimal operator sets

## Future Work

1. **Adaptive Libraries**: Start with minimal library and gradually add functions
2. **Function Importance**: Analyze which functions are most/least useful
3. **Problem-Specific Libraries**: Design libraries based on problem characteristics
4. **Multi-Objective**: Balance accuracy vs. library size 