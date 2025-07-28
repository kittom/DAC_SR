# Minimal PySR configuration for OneMax problem
# Ground truth: sqrt(x1/(x1-x2))
# Required functions: +, -, /, sqrt

BINARY_OPERATORS = ["+", "-", "/"]
UNARY_OPERATORS = ["sqrt"]

# Note: No power operator (^) needed for OneMax - sqrt is sufficient
# No trigonometric functions needed
# No exponential/logarithmic functions needed
# No absolute value needed - sqrt handles positive values 