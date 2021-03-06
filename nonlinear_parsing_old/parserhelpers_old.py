import sympy as sym
import numpy as np

from scipy.sparse.linalg import norm
from scipy.sparse import csc_matrix, identity, triu, csr_matrix
from sksparse.cholmod import cholesky
import matplotlib.pylab as plt

import math
import itertools
from operator import add, itemgetter

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
            C[j, i] = 1
    return  # Just updating C, no return value


# Main function that generates clique structure from matrix of codependencies
# A = L * L' reconstructs the original matrix
def cliquesFromSpMatD(C):
    factor = cholesky(C)   # Requires creating this factor object
    L = factor.L()         # Retrieve lower triangular matrix
    L = csc_matrix([[1, 0, 0], [1, 1, 0], [0, 1, 1]]) # Temporary

    cliques = L.copy().tocsr()  # Populate same L matrix, except with ones
    cliques.data.fill(1)

    x_index = [i for i in range(C.shape[0])]     # Array of indices
    n = len(x_index)                             # Length of problem statement
    remainIndex = [0]                            # Start a list of remaining indices

    # Iterating n-1 times (seems to be skipping the first row and column and moving diagonally)
    for i in range(1, n):
        idx = np.array([j for j in range(i, n)])     # Array of ROW indices to check from i until the end
        one = np.argwhere(cliques[i:, i])  # Returns indices where nonzero values are found (RELATIVE TO SUBMATRIX)
        parsedOne = [i[0] for i in one]    # Parse to MatLab equivalent one
        numOne = len(parsedOne)            # Number of ones

        cliqueResult = cliques[idx[parsedOne], remainIndex].sum()  # Complicated result to retrieve result

        # Check if any of the cliqueResult values is equal to the number of ones
        if not isinstance(cliqueResult, list): cliqueResult = [cliqueResult]  # Make iterable if scalar
        if next((i for i in cliqueResult if i == numOne), None) is None:  
            remainIndex.append(i)   # Add to the remaining indices

    # Gather clique information
    cliqueSet = cliques[:, remainIndex]
    noCliques = len(remainIndex)
    elemInfo = np.argwhere(cliqueSet)
    elem = [i[0] for i in elemInfo]
    noElem = np.squeeze(np.asarray(cliqueSet.sum(axis=0)))
    maxC = max(noElem)
    minC = min(noElem)

    # Create upper triangular matrix
    cliquesUpper = cliques[:, remainIndex]
    cliquesUpper = triu(cliquesUpper * cliquesUpper.transpose())

    # Gather non-zero entry indices in matrix
    nonZeroes = np.argwhere(cliquesUpper)
    nonZeroes = sorted(nonZeroes, key=itemgetter(0, 1))

    s = len(nonZeroes)
    rows = [i[0] for i in nonZeroes]
    cols = [i[1] for i in nonZeroes]
    data = [i for i in range(1, s+1)]

    idxMatrix = csr_matrix((data, (rows, cols)), shape=(n, n))

    cliqueSet = [np.array(elem[:noElem[0]])]
    for i in range(1, noCliques):
        idx = sum(noElem[:i])
        idxs = [(idx + i) for i in range(noElem[i])] 
        cliqueSet.append(np.array(elem)[idxs])

    return 1

    

# Main function for gathering cliques
def corrSparsityCliques(x, obj, constraints):
    n = len(x)                     # Independent variable vector size
    C = identity(n, format='lil')  # Initialising appropriate identity matrix (lil_matrix format)

    # Retrieve cross-terms of objective function and update C matrix
    objectiveCrossDependencies = getObjectiveCrossDependencies(obj, x)
    updateCWithCrossDependencies(objectiveCrossDependencies, C)

    # Retrieve co-dependent terms for every constraint and update C matrix
    for constraint in constraints:
        constraintCrossDependencies = getConstraintCodependencies(constraint, x)
        updateCWithCrossDependencies(constraintCrossDependencies, C)
   
    C = csc_matrix(C)                                 # Convert into CSC structure, most efficient for next steps
    C = C + norm(C, ord=1)*identity(n, format='csc')  # Ensure strict diagonal dominance for C      

    cliqueStructure = cliquesFromSpMatD(C)  # Large function that will generate clique structure

    return cliqueStructure

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


