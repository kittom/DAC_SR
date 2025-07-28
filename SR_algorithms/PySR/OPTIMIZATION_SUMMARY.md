# PySR Optimization Summary

## Overview
This document summarizes the optimizations applied to PySR scripts for improved speed and fairness in the DAC_SR experiment framework.

## Key Optimizations Applied

### 1. **Fairness Constraints**
- **Single CPU**: `procs=1` ensures fair comparison with other algorithms
- **Consistent resources**: All runs use the same computational resources
- **No parallel advantage**: Prevents PySR from using multiple cores unfairly

### 2. **Speed Optimizations (from PySR Tuning Guide)**

#### **Turbo Mode**
- `turbo=True`: Enables advanced loop vectorization for ~20% speedup
- Experimental but stable feature recommended by PySR developers

#### **Optimized Parameters**
- **Population Size**: 50 (conservative for single CPU)
- **Cycles per Iteration**: 1000 (balanced for single CPU vs cluster settings)
- **Parsimony**: `noise_threshold / 8.0` (optimal balance between exploration and convergence)
- **Weight Optimization**: 0.001 (ensures frequent optimization)

#### **Adaptive Settings**
- **Batching**: Enabled for datasets > 1000 samples
- **Population Reduction**: Smaller population (30) for large datasets
- **Precision**: 32-bit for speed vs 64-bit for precision

### 3. **Operator Optimization**

#### **Reduced Redundancy**
- Removed redundant operators: `tan`, `tanh` (can be expressed with `sin`, `cos`)
- Kept essential operators only
- Applied constraints to prevent redundant expressions

#### **Constraints**
- **Power Constraints**: `{"pow": (9, 1)}` - only variable/constant exponents
- **Nested Trig**: `{"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}` - no nested trig functions

#### **Complexity Settings**
- Higher complexity for complex operators (`pow: 3`, `sin/cos/exp/log: 2`)
- Lower complexity for simple operators (`sqrt: 1.5`, `abs: 1`)

### 4. **Minimal Library Configurations**

#### **OneMax Problem**
- Operators: `["+", "-", "/", "sqrt"]`
- Ground truth: `sqrt(x1/(x1-x2))`
- No power, trig, or exponential functions needed

#### **LeadingOnes Problem**
- Operators: `["+", "/"]`
- Ground truth: `x1/(x2 + 1)`
- Minimal set - no subtraction, power, or unary operators

#### **PSA-CMA-ES Problem**
- Operators: `["*", "-", "/", "exp"]`
- Ground truth: `x1 * exp(x2 * (x5 - (x3 / x4)))`
- No addition, power, or trig functions needed

### 5. **Robustness Features**

#### **Exploration Settings**
- `adaptive_parsimony_scaling=100`: Helps with exploration
- `warmup_maxsize_by=0.5`: Starts with smaller expressions
- `maxsize=25`: Increased from 20 for better exploration

#### **Loss Function**
- `loss="L2DistLoss()"`: Standard L2 loss for regression
- Consistent with other algorithms

#### **Error Handling**
- Robust convergence checking
- Graceful handling of PySR errors
- Cleanup of temporary directories

## Performance Expectations

### **Speed Improvements**
- **Turbo Mode**: ~20% speedup
- **Optimized Parameters**: ~30-50% speedup
- **Reduced Operators**: ~20-40% speedup (depending on problem)
- **Total Expected**: 50-80% speedup vs default settings

### **Fairness Guarantees**
- Single CPU usage ensures fair comparison
- Consistent resource allocation
- No parallel processing advantage

### **Quality Maintenance**
- Minimal library configurations maintain solution quality
- Constraints prevent overfitting to redundant expressions
- Adaptive settings ensure good exploration

## Usage

### **Standard PySR (Full Library)**
```bash
python run_pysr.py dataset.csv --noise 1e-6 --max-iterations 100
```

### **Minimal Library PySR**
```bash
python run_pysr_lib.py dataset.csv one_max --noise 1e-6 --max-iterations 100
```

## Monitoring

The scripts now provide detailed output including:
- Dataset size and features
- Parsimony, population, and cycle settings
- Batching and turbo mode status
- Convergence progress
- Final performance metrics

This ensures transparency and allows for monitoring of the optimization effectiveness. 