import sympy as sym
import numpy as np
import math




# Gather exponent bases (TODO:)
def getExponents(obj, x):
    """
    For every 'monomial/term' inside the objective function or constraint, check to see if there is a dependent on each of the variables

    Each row represents a monomial in the objectiove function
    Each column is one of the independent variables x_i
    """

    return 1

# Main function for gathering cliques
def corrSparsityCliques(x, obj, constraints):
    n = len(x)          # Independent variable vector size
    C = np.identity(n)  # Initialising appropriate identity matrix

    # Get correlative sparsity matrix of objective function
    exponents = getExponents(obj, x)

    return 1

# Main function that will return the relaxed optimisation problem
def compileParseMoment(x, objectiveFunction, equalityConstraints, inequalityConstraints, omega=None):
    # Useful constants
    n = len(x)
    xID = [x for x in range(n)]

    # Find maximum degree of each constraint and objective function
    equalityConstraintsDegrees = [sym.Poly(constraint).total_degree() for constraint in equalityConstraints]
    inequalityConstraintsDegrees = [sym.Poly(constraint).total_degree() for constraint in inequalityConstraints]
    objectiveFunctionDegree = sym.Poly(objectiveFunction).total_degree()
    maxDegree = math.ceil(max(max(equalityConstraintsDegrees), max(inequalityConstraintsDegrees), objectiveFunctionDegree) / 2)

    # Corect omega if required
    omega = maxDegree if omega is None or omega < maxDegree else omega

    # Find cliques
    CD = corrSparsityCliques(x, objectiveFunction, [equalityConstraints, inequalityConstraints])

    # TODO: Continuing from here

    return (1, 1, 1, 1)


