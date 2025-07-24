# Minimal PySR configuration for PSA-CMA-ES problem
# Ground truth: x1 * exp(x2 * (x5 - (x3 / x4)))
# Required functions: *, -, /, exp

BINARY_OPERATORS = ["*", "-", "/"]
UNARY_OPERATORS = ["exp"]

# Note: No addition (+) needed for PSA-CMA-ES
# No power operator (^) needed
# No trigonometric functions needed
# No square root needed
# No logarithm needed 