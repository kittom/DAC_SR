Symbolic Regression Experiment Fairness Standards
================================================

This document outlines the standards and practices for ensuring fair comparison between different symbolic regression algorithms in our experiments.

## 1. Data Generation Standards

### 1.1 Dataset Structure
- All datasets must follow the same format: features (x1, x2, ..., xn) followed by target variable (y)
- No headers in CSV files to ensure consistent parsing
- Ground truth equations must be stored in results.csv files with standardized variable naming (x1, x2, ..., xn)

### 1.2 Variable Naming Convention
- Input variables: x1, x2, x3, ..., xn (where n is the number of input features)
- Target variable: y (not included in ground truth equations)
- Ground truth equations must use x1, x2, ..., xn notation for consistency

### 1.3 Data Quality
- All datasets must be generated using the same random seeds for reproducibility
- Data must be properly scaled and normalized where appropriate
- No data leakage between training and evaluation sets

## 2. Algorithm Configuration Standards

### 2.1 Mathematical Function Libraries
- **Standard Library**: All algorithms must support the same core mathematical functions
- **Required Functions**: 
  - Binary: `+`, `-`, `*`, `/`, `^` (or `**`)
  - Unary: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `exp`, `log`, `sqrt`, `abs`, `tanh`
  - Special: `x^2`, `x^3`, `1/x`
- **Implementation**: See `mathematical_libraries_standard.md` for detailed specifications
- **Fairness**: No algorithm should be disadvantaged by missing mathematical functions

### 2.2 Loss Functions
- **Principle**: Use each algorithm's native loss function as designed
- **Rationale**: Changing loss functions would be unfair as algorithms are optimized for their specific loss metrics
- **Implementation**: Accept that different algorithms use different internal loss functions (MSE, NMSE, custom rewards)

### 2.3 Stopping Criteria
- **Standard**: Use convergence-based stopping when possible, with noise-level threshold
- **Threshold**: Set convergence threshold equal to the noise level in the data
- **Minimum Iterations**: Prevent premature stopping (typically 10-20 iterations)
- **Maximum Iterations**: Prevent infinite loops (typically 100-200 iterations)

### 2.4 Computational Budget
- **Time Limits**: Set reasonable timeouts for all algorithms
- **Memory Limits**: Ensure all algorithms have similar memory constraints
- **CPU/GPU Usage**: Use consistent hardware configurations

## 3. Evaluation Standards

### 3.1 Metrics
- **Primary Metric**: Normalized Mean Square Error (NMSE) for final comparison
- **Secondary Metrics**: Mean Absolute Error (MAE), R-squared, complexity measures
- **Formula**: NMSE = sqrt(mean((y_true - y_pred)^2) / mean(y_true^2))

### 3.2 Equation Quality Assessment
- **Parsability**: All equations must be parseable by sympy
- **Complexity**: Track equation complexity (number of operations, depth)
- **Interpretability**: Assess human readability and mathematical correctness

### 3.3 Statistical Significance
- **Multiple Runs**: Perform multiple runs with different random seeds
- **Confidence Intervals**: Report confidence intervals for performance metrics
- **Statistical Tests**: Use appropriate statistical tests for comparing algorithms

## 4. Implementation Standards

### 4.1 Code Organization
- **Modular Design**: Separate data generation, algorithm execution, and evaluation
- **Reproducibility**: All experiments must be reproducible with provided scripts
- **Documentation**: Clear documentation of all parameters and configurations

### 4.2 Error Handling
- **Graceful Failures**: Handle algorithm failures without stopping entire experiment
- **Logging**: Comprehensive logging of all operations and decisions
- **Recovery**: Ability to resume experiments from failure points

### 4.3 Results Storage
- **Standardized Format**: All results stored in consistent CSV format
- **Metadata**: Include experiment parameters, timestamps, and configuration details
- **Version Control**: Track changes to algorithms and configurations

## 5. Fairness Considerations

### 5.1 Algorithm-Specific Limitations
- **Acknowledge Differences**: Different algorithms have different strengths and limitations
- **Appropriate Use Cases**: Match algorithms to problem characteristics
- **Transparent Reporting**: Clearly report what each algorithm can and cannot do

### 5.2 Baseline Comparisons
- **Linear Regression**: Always include linear regression as a baseline
- **Random Baselines**: Include random or naive baselines where appropriate
- **State-of-the-Art**: Compare against published results when available

### 5.3 Interpretability vs Performance
- **Trade-offs**: Acknowledge that simpler equations may have slightly worse performance
- **Multi-objective**: Consider both accuracy and interpretability in evaluation
- **Human Assessment**: Include human evaluation of equation quality where possible

## 6. Reporting Standards

### 6.1 Experiment Documentation
- **Complete Configuration**: Document all parameters for each algorithm
- **Hardware Details**: Report computational resources used
- **Timing Information**: Include wall-clock time and CPU time

### 6.2 Results Presentation
- **Clear Tables**: Present results in clear, comparable tables
- **Visualizations**: Include appropriate plots and graphs
- **Statistical Summary**: Provide mean, standard deviation, and confidence intervals

### 6.3 Limitations and Caveats
- **Honest Reporting**: Clearly state limitations of each approach
- **Failure Cases**: Document and analyze cases where algorithms fail
- **Generalization**: Discuss how well results generalize to other problems

## 7. Continuous Improvement

### 7.1 Algorithm Updates
- **Version Tracking**: Track versions of all algorithms used
- **Update Policy**: Establish policy for when to update algorithms
- **Backward Compatibility**: Ensure new versions don't break existing experiments

### 7.2 Standard Evolution
- **Regular Review**: Periodically review and update these standards
- **Community Input**: Incorporate feedback from the research community
- **Best Practices**: Stay current with best practices in the field

## 8. Usage Guidelines

### 8.1 For New Experiments
1. Review this standards document before starting
2. Ensure all algorithms are configured according to these standards
3. Document any deviations and their rationale
4. Use the provided evaluation scripts and templates

### 8.2 For Algorithm Comparisons
1. Use the same datasets and evaluation metrics
2. Ensure fair computational budgets
3. Report results in standardized format
4. Include appropriate statistical analysis

### 8.3 For Result Reporting
1. Follow the reporting standards outlined above
2. Include all relevant metadata and configuration details
3. Be transparent about limitations and assumptions
4. Provide code and data for reproducibility

---

This document should be reviewed and updated regularly to ensure continued fairness and rigor in symbolic regression experiments. 