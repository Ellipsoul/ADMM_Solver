import sympy as sym
import numpy as np
from helpers import compileParseMoment

# Define problem size and initialise symbolic variables
SIZE = 3
x = sym.Matrix([sym.Symbol(f'x{i}') for i in range(SIZE)])

# Define a polynomial objective function
objectiveFunction = x.dot(x) - sum(x)

# Define some equality constraints
equalityConstraints = [ 
    x[0] - x[1]**2,
    x[1] * x[2],
    1 - x[1] - x[1]**2 + x[2]
]
# Define some inequality constraints
inequalityConstraints = [ 
    1 - 4*x[0]**2,
    1 - x[0]**2 - x[1]**2,
    1 - x[1]**2 - x[2]**2
]
OMEGA = 3  # Define the degree of relaxation

numpyX = np.transpose(np.array(x))[0]  # Convert independent variable matrix to numpy array

(At, b, c, K) = compileParseMoment(numpyX, objectiveFunction, equalityConstraints, inequalityConstraints, OMEGA)
