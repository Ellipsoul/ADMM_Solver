import sympy as sym
import numpy as np
import math
import itertools
from operator import add

# Gather all cross-term dependencies for objective function
def getObjectiveCrossDependencies(obj, x):
    """
    For every monomial/term inside the function f, retrive the variable dependencies
    Return a list of of dependency lists
    """
    terms = sym.Poly(obj, *x).terms()             # Retrive all terms from the polynomial (pass x vars as args)
    monomials = [term[0] for term in terms]       # Extract matrix representation of monomials from terms

    crossDependencies = []                        # Initialise array of co-dependent terms
    for monomial in monomials:                    # Iterate through each monomial array
        # Create an array where non-zero entries represents a cross-term
        crossDependency = [i for i, value in enumerate(monomial) if value]
        # Append to codependencies if it's a cross-term
        if len(crossDependency) > 1:
            crossDependencies.append(crossDependency)
    
    return crossDependencies


# Gather all codependencies for constraint functions
def getConstraintCodependencies(constraint, x):
    """
    Considering the polynomial as a whole, retrieve all independent variable dependencies on the polynomial
    Return a single list of dependencies
    """
    terms = sym.Poly(constraint, *x).terms()                    # Retrive all terms from the polynomial
    monomials = [np.abs(term[0]) for term in terms]             # Extract matrix representation of monomials from terms
    aggregaredDependencies = [sum(x) for x in zip(*monomials)]  # Collapse polynomial codependencies (elementwise sum)

    # Gather codependency list (place in nested lists for compatibility with updateCWithCrossDependencies)
    codependencies = [[i for i, value in enumerate(aggregaredDependencies) if value]]
    return codependencies


# Update the C matrix using the cross dependencies
def updateCWithCrossDependencies(crossDependencies, C):
    for crossDependency in crossDependencies:                       # Iterate through all dependencies
        combinations = itertools.combinations(crossDependency, 2)   # Generate possible combinations
        for combination in combinations:
            i = combination[0]
            j = combination[1]
            C[i, j] = 1
            C[j, i] = 1  # TODO: What is the MATLAB script actually doing??
    return  # Just updating C, no return value


# Main function that generates clique structure from matrix of codependencies
def cliquesFromSpMatD(C):
    # TODO: Continue from here

    return 1

# Main function for gathering cliques
def corrSparsityCliques(x, obj, constraints):
    n = len(x)          # Independent variable vector size
    C = np.identity(n)  # Initialising appropriate identity matrix

    # Retrieve cross-terms of objective function and update C matrix
    objectiveCrossDependencies = getObjectiveCrossDependencies(obj, x)
    updateCWithCrossDependencies(objectiveCrossDependencies, C)

    # Retrieve co-dependent terms for every constraint and update C matrix
    for constraint in constraints:
        constraintCrossDependencies = getConstraintCodependencies(constraint, x)
        updateCWithCrossDependencies(constraintCrossDependencies, C)
    
    cliques = cliquesFromSpMatD(C)  # Large function that will generate clique structure

    return cliques

# Main function that will return the relaxed optimisation problem
def compileParseMoment(x, objectiveFunction, equalityConstraints, inequalityConstraints, omega=None):
    # Useful constants
    n = len(x)
    xID = [x for x in range(n)]

    # Find maximum degree of each constraint and objective function
    equalityConstraintsDegrees = [sym.Poly(constraint, *x).total_degree() for constraint in equalityConstraints]
    inequalityConstraintsDegrees = [sym.Poly(constraint, *x).total_degree() for constraint in inequalityConstraints]
    objectiveFunctionDegree = sym.Poly(objectiveFunction, *x).total_degree()
    maxDegree = math.ceil(max(max(equalityConstraintsDegrees), max(inequalityConstraintsDegrees), objectiveFunctionDegree) / 2)

    # Corect omega if required
    omega = maxDegree if omega is None or omega < maxDegree else omega

    # Find cliques
    CD = corrSparsityCliques(x, objectiveFunction, equalityConstraints + inequalityConstraints)

    # TODO: Continuing from here

    return (1, 1, 1, 1)


