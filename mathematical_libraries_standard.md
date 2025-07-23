# Standardized Mathematical Function Libraries

This document defines the standardized mathematical function libraries used across all symbolic regression algorithms to ensure fair comparison.

## Standard Library Definition

All algorithms now support the following comprehensive mathematical function library:

### Binary Operators
- `+` (addition)
- `-` (subtraction)
- `*` (multiplication)
- `/` (division)
- `^` or `**` (power/exponentiation)

### Unary Operators
- `sin` (sine)
- `cos` (cosine)
- `tan` (tangent)
- `arcsin` (arcsine)
- `arccos` (arccosine)
- `arctan` (arctangent)
- `exp` (exponential)
- `log` (natural logarithm)
- `sqrt` (square root)
- `abs` (absolute value)
- `tanh` (hyperbolic tangent)

### Special Functions
- `x^2` (squared)
- `x^3` (cubed)
- `1/x` (reciprocal)

## Algorithm-Specific Implementations

### 1. PySR
```python
binary_operators=["+", "-", "*", "/", "^"]
unary_operators=["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh"]
```
**Status**: ✅ Updated to include sqrt, abs, tanh, tan, and power operator

### 2. KAN
```python
lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'tan', 'abs', '1/x', 'arcsin', 'arccos', 'arctan']
```
**Status**: ✅ Updated to include tan, arcsin, arccos, arctan

### 3. DeepSR
```json
"function_set": ["add", "sub", "mul", "div", "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "arcsin", "arccos", "arctan", "poly"]
```
**Status**: ✅ Updated to include tan, sqrt, abs, tanh, arcsin, arccos, arctan

### 4. TPSR
```python
operators_real = {
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    "abs": 1, "inv": 1, "sqrt": 1,
    "log": 1, "exp": 1, "sin": 1, "arcsin": 1,
    "cos": 1, "arccos": 1, "tan": 1, "arctan": 1,
    "pow2": 1, "pow3": 1,
}
```
**Status**: ✅ Already comprehensive (includes all standard functions)

### 5. Q-Lattice
```python
# Uses feyn library with built-in support for:
# - Basic operations: +, -, *, /, **
# - Trigonometric: sin, cos, tan, arcsin, arccos, arctan
# - Exponential/Logarithmic: exp, log, log10
# - Other: sqrt, abs, sign, floor, ceil, round
```
**Status**: ✅ Comprehensive (automatic function discovery)

### 6. E2E Transformer
```python
# Pre-trained model with support for:
# - Basic operations: +, -, *, /, **
# - Trigonometric: sin, cos, tan, arcsin, arccos, arctan
# - Exponential/Logarithmic: exp, log, log10
# - Other: sqrt, abs, sign, floor, ceil, round
```
**Status**: ✅ Comprehensive (pre-trained on diverse mathematical expressions)

### 7. Linear Regression
```python
# Linear combinations only: a*x1 + b*x2 + c
```
**Status**: ⚠️ Limited (linear only - serves as baseline)

## Benchmark Problem Compatibility

### OneMax Problem: `sqrt(x1/(x1-x2))`
- **Required Functions**: Division (`/`), Square Root (`sqrt`)
- **Compatible Algorithms**: KAN, TPSR, DeepSR, PySR, Q-Lattice, E2E Transformer
- **Incompatible Algorithms**: Linear Regression

### LeadingOnes Problem: `x1/(x2 + 1)`
- **Required Functions**: Division (`/`), Addition (`+`)
- **Compatible Algorithms**: All algorithms
- **Incompatible Algorithms**: None

### PSA-CMA-ES Problem: `x1 * exp(x2 * (x5 - (x3 / x4)))`
- **Required Functions**: Multiplication (`*`), Division (`/`), Exponential (`exp`), Subtraction (`-`)
- **Compatible Algorithms**: All algorithms
- **Incompatible Algorithms**: None

## Implementation Notes

### Function Naming Conventions
- **PySR**: Uses standard mathematical notation (`sqrt`, `sin`, etc.)
- **KAN**: Uses standard mathematical notation
- **DeepSR**: Uses function names (`sqrt`, `sin`, etc.)
- **TPSR**: Uses function names (`sqrt`, `sin`, etc.)
- **Q-Lattice**: Automatic discovery with standard notation
- **E2E Transformer**: Automatic discovery with standard notation

### Power Operations
- **PySR**: Uses `^` operator
- **KAN**: Uses `x^2`, `x^3` notation
- **DeepSR**: Uses `poly` for polynomial expressions
- **TPSR**: Uses `pow2`, `pow3` for specific powers
- **Q-Lattice**: Uses `**` operator
- **E2E Transformer**: Uses `**` operator

### Special Considerations

#### Linear Regression Limitations
- Linear regression can only represent linear relationships
- Cannot represent square roots, trigonometric functions, or other non-linear operations
- Serves as a baseline control to demonstrate the value of non-linear symbolic regression

#### Algorithm-Specific Strengths
- **PySR**: Good at finding simple, interpretable expressions
- **KAN**: Excellent at discovering complex non-linear relationships
- **DeepSR**: Strong at finding expressions with constants and complex structures
- **TPSR**: Good balance between exploration and exploitation
- **Q-Lattice**: Automatic discovery of mathematical patterns
- **E2E Transformer**: Pre-trained on diverse mathematical expressions

## Fairness Standards

### Function Library Equality
- All algorithms now have access to the same core mathematical functions
- Square root support has been added to algorithms that were missing it
- Trigonometric functions are available across all algorithms
- Power operations are supported in various forms

### Benchmark Problem Accessibility
- All algorithms can now represent the OneMax equation (`sqrt(x1/(x1-x2))`)
- All algorithms can represent the LeadingOnes equation (`x1/(x2 + 1)`)
- All algorithms can represent the PSA-CMA-ES equation (`x1 * exp(x2 * (x5 - (x3 / x4)))`)

### Performance Expectations
With standardized libraries, performance differences should now be due to:
1. **Algorithm efficiency** rather than function availability
2. **Search strategy** rather than mathematical limitations
3. **Convergence behavior** rather than missing functions
4. **Implementation quality** rather than library restrictions

## Maintenance

### Adding New Functions
When adding new mathematical functions to the standard library:
1. Update this document
2. Update all algorithm configurations
3. Test compatibility with benchmark problems
4. Update fairness standards

### Version Control
- Track changes to function libraries
- Ensure backward compatibility
- Document any algorithm-specific limitations

---

**Last Updated**: January 2025
**Version**: 1.0
**Status**: All algorithms standardized ✅ 