import scipy
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, dia_matrix, identity, csc_matrix, lil_matrix
from scipy.sparse.linalg import inv
from sksparse.cholmod import cholesky
import math

import numpy as np
import matplotlib.pylab as plt
import networkx as nx

# Overarching Solution Structure Containing all cliques and the original data
class SolutionStructure:
    def __init__(self, At, b, c, K, options):
        checkInputs(At, b, c, K)  # Check inputs (and importantly format K)

        # Original statement components
        self.At = At
        self.b = b
        self.bt = self.b.transpose()
        self.c = c
        self.K = K

        self.options = options                                       # Options
        self.cliqueComponents = detectCliques(At, b, c, K, options)  # Detect the cliques!
        self.ncliques = len(self.cliqueComponents)

        self.time = CPUTime()                          # Tracking time for algorithm

        self.objectiveCost = None                      # Initialise cost (will be updated at each iteration)
        self.iterationPrimalResidual = float("inf")    # Initialise iteration max primal residual
        self.iterationDualResidual = float("inf")      # Initialise iteration max dual residual
        self.y = None                                  # Initialise solution variable

        # Gather lefthand matrix inverse (for y minimisation step)
        lefthandMatrixSum = sum(clique.L for clique in self.cliqueComponents)  # Sum lefthand diagonal matrix
        diagReciprocals = np.reciprocal(lefthandMatrixSum.data)                # Gather reciprocals of diagonal
        nDiag = len(diagReciprocals)
        self.Linv = scipy.sparse.dia_matrix((diagReciprocals, [0]), shape=(nDiag, nDiag))  # Create matrix inverse


# Class to wrap all variables within each subproblem
class CliqueComponent:
    def __init__(self, At, b, c, K, P, options):
        # Problem statement components
        self.At = At
        self.A = self.At.transpose()   # Store the transpose since transposing is expensive
        self.b = b
        self.c = c
        self.K = K
        self.P = P
        self.Pt = self.P.transpose()   # Store the transpose since transposing is expensive
        
        # Useful constants
        self.rho = options.rho
        self.sigma = options.sigma
        self.lamb = options.lamb

        # Lagrange multiplier vectors to be updated at each cycle
        self.zeta = np.ones(shape=(self.b.shape[0], 1))
        self.eta = np.ones(shape=(self.c.shape[0], 1))

        # Initialise local s vector containing extracted components on the full y vector
        self.s = np.zeros(shape=(self.b.shape[0], 1))
        
        # Initialise local z cost vector for ensuring problem remains conic
        zeroCones = np.zeros(shape=(K["f"], 1))                              # Equality: 0s
        nnOrthants = np.ones(shape=(K["l"], 1))                              # Inequality: 1s
        PSDs = [np.identity(size).reshape((size**2, 1)) for size in K["s"]]  # PSD: identity
        self.z = vstack([zeroCones, nnOrthants, *PSDs])                      # Stack up the vectors for constraints

        # Generate matrix: L = rho_i*Pt*P (static if rho kept constant)
        self.L = self.rho * self.Pt * self.P                      
        # Generate matrix: R = rho * I + sigma * A * At (static if rho and sigma kept constant), and its inverse
        self.R = csc_matrix(self.rho * identity(len(self.s)) + self.sigma * (self.A * self.At))
        # Cholesky factorisation for most efficient system of equations solution
        self.KKt = cholesky(self.R, 0)

        self.yUpdateVector = self.Pt * (self.zeta + self.rho * self.s)    # Initialise first y updating value

        # Initialise initial primal and dual residuals

        # Local primal residual (c - At*s - z)
        # The other residual (s - P*y) will be calculated globally since it requires access to the global y vector
        self.primalResidualLocal = np.linalg.norm(self.c - self.At*self.s - self.z)
        # First Dual Residual: -lambda * b - sum(Pt * zeta^k+1) = rho * Pt * (s^k - s^k+1)  
        # Must be summed across all cliques 
        self.dualResidualOne = float("inf")
        # Second Dual Residual: sigma * At (s^k+1 - s^k)
        # This one is local, so there is one for each clique
        self.dualResidualTwo = float("inf")


    # Generate righthand local value for minimising with respect to y
    def updateYUpdateVector(self):
        self.yUpdateVector = self.Pt * (self.zeta + self.rho * self.s)

    # Update L diagonal matrix (if code needs to be extended to vary rho)
    def updateLMatrix(self):
        self.L = self.rho * (self.Pt * self.P)    # Update L (if rho is programmed to be dynamic)

    # Update R matrix and corresponding inverse (in use only when rho and/or sigma is dynamic)
    def updateRMatrix(self):
        self.R = csc_matrix(self.rho * identity(len(self.s)) + self.sigma * (self.A * self.At))
        self.KKt = cholesky(self.R, 0)


# Class of options
class Options:
    def __init__(self, rho=10, sigma=10, lamb=0.5, relTol=1.0e-06, maxIter=1000, dispIter=50):
        self.rho = rho                      # Penalty parameter 1
        self.sigma = sigma                  # Penalty parameter 2
        self.lamb = lamb                    # Weighting between large vector optimisation and clique optimisation

        self.relTol = relTol                # Relative tolerance
        self.maxIter = maxIter              # Maximum number of iterations
        self.dispIter = dispIter            # Iterations for which to print


# Class to store CPU times
class CPUTime:
    def __init__(self, start=0):
        self.start = start
        self.init = 0.0
        self.cliqueDetection = 0.0
        self.elapsed = 0.0

        self.updateY = 0.0
        self.updateZ = 0.0
        self.updateS = 0.0

        self.updateLagrangeMultipliers = 0.0
        self.updateResiduals = 0.0

        self.calculateCost = 0.0

        self.temp = 0.0  # Use this to test time specific parts of the solver


# Helper function to vectorise a matrix, taking in a sparse matrix
def vectoriseMatrix(M):
    n = M.shape[0]**2               # Length of long vector
    return M.reshape((1, n))        # Reshape matrix in place with no return value


# Helper function for converting a vector back into a square matrix
def matriciseVector(v):
    n = int(math.sqrt(v.shape[0]))
    return v.reshape((n, n))


# Large function for detecting and splitting cliques
def detectCliques(At, b, c, K, options):
    
    nConstraints = K.f + K.l + len(K.s)   # Helper to track number of constraints
    nCols = At.shape[1]                   # Useful later for P matrix construction

    # Initialise array of constraint components
    constraints = [None for _ in range(nConstraints)]

    # Populate with equality constraints
    for i in range(K.f): 
        constraint = At[i, :]                                 # Extract constraint
        nonZeroIndices = set(np.nonzero(constraint)[1])       # Generate set of nonzero indices in constraint
        cConstraint = c[i]                                    # Retrieve c subvector

        constraints[i] = [constraint, nonZeroIndices, cConstraint, "eq"]   # Gather and store info
    
    # Populate with inequality constraints
    for i in range(K.f, K.f + K.l): 
        constraint = At[i, :]                                 # Extract constraint 
        nonZeroIndices = set(np.nonzero(constraint)[1])       # Generate set of nonzero indices in constraint
        cConstraint = c[i]                                    # Retrieve c subvector

        constraints[i] = [constraint, nonZeroIndices, cConstraint, "ineq"]  # Gather and store info

    # Replace all non-zero data with just ones
    AtOnes = At.copy().tocsr()
    AtOnes.data.fill(1)

    # K.f - Fixed, K.l - Linear, K.s[] - Semidefinite
    # Collapse single matrix constrants into single row
    AtHead = AtOnes[:K.f+K.l, :]   # Matrix of equality + inequality constraints to be stacked with collapsed PSDs
    collapsedRows = []
    
    currentIdx = K.f + K.l
    # Iterate through all PSD constraints
    ptr = K.f + K.l
    for matrixSize in K.s:
        rowsToExtract = matrixSize ** 2
        psdConstraint = At[currentIdx: currentIdx+rowsToExtract, :]             # Retrive the matrix subset
        cConstraint = c[currentIdx: currentIdx+rowsToExtract]                   # Retrive c vector subset

        collapsedRow = AtOnes[currentIdx: currentIdx+rowsToExtract, :]
        collapsedRow = collapsedRow.sum(axis=0)                                 # Collapse into a single vector
        collapsedRows.append(collapsedRow)                                      # Store the collapsed row

        nonZeroIndices = set(np.nonzero(psdConstraint)[1])                      # Generate set of nonzero indices in constraint
        constraints[ptr] = [psdConstraint, nonZeroIndices, cConstraint, "psd"]  # Gather and store info
        ptr += 1                                                                # Increment pointer

        currentIdx += rowsToExtract                                             # Increment row counter

    AtCollapsed = vstack([AtHead] + collapsedRows)                              # Stack rows together

    # Find cliques
    S = AtCollapsed.transpose() * AtCollapsed              # Generate matrix of codependencies
    G = nx.Graph(S)                                        # Initialise NetworkX graph
    cliques = list(nx.algorithms.find_cliques(G))          # Retrieve cliques

    for i in range(len(cliques)): cliques[i] = np.sort(cliques[i])

    # Gather the related At, b, c, K and P arrays for each detected clique

    # Initiatlise output values
    At_sparse = [None for i in range(len(cliques))]
    b_sparse = [None for i in range(len(cliques))]
    c_sparse = [None for i in range(len(cliques))]
    K_sparse = [None for i in range(len(cliques))]
    P_sparse = [None for i in range(len(cliques))]

    constraints = np.array(constraints)

    # Iterate through each cliques
    for i in range(len(cliques)):
        currentClique = cliques[i]           # Extract current clique
        
        constraintIndices = []
        # Iterate to find constraints that belong in this clique 
        for j in range(len(constraints)):
            currentConstraint = constraints[j]                 # Grab current constraint in question
            constraintDependencies = currentConstraint[1]      # Retrieve dependencies

            # Add to the indices list if constraint is included in the clique
            # IMPORTANT: Check if all dependencies are in the current clique, add the constraint if satisfied
            constraintIsInClique = all(dep in currentClique for dep in constraintDependencies)
            if constraintIsInClique: constraintIndices.append(j)

        constraintsInClique = constraints[constraintIndices]    # Define the constraints in the clique

        c_sparse[i] = vstack(constraintsInClique[:, 2])         # Stack up c vector

        # Stack A matrix, removing columns that only contain zeroes
        At_sparseFull = vstack(constraintsInClique[:, 0])        # Stack up sparse A matrix
        indices = np.nonzero(At_sparseFull)                      # Find indices where there are nonzero values
        nonZeroColumns = sorted(set(indices[1]))                 # Grab nonzero columns and sort
        At_sparse[i] = At_sparseFull.tocsc()[:, nonZeroColumns]  # Finally, select only nonzero columns and store
        
        # Initialise and populate K structure for each clique
        K_sparse[i] = {"f": 0, "l": 0, "s": []}
        for constraintInClique in constraintsInClique:          # Check the case type and increment
            if constraintInClique[3] == "eq": K_sparse[i]["f"] += 1
            elif constraintInClique[3] == "ineq": K_sparse[i]["l"] += 1
            elif constraintInClique[3] == "psd": 
                psdSize = int(math.sqrt(constraintInClique[0].shape[0]))   # Define PSD matrix size
                K_sparse[i]["s"].append(psdSize)                           # Add

        # Initialise and populate P matrix
        P_sparse[i] = lil_matrix((len(currentClique), nCols))                       # Initialise sparse directly
        for k in range(len(currentClique)): P_sparse[i][k, currentClique[k]] = 1    # Populate with appropriate ones

    # Populate sparse b vectors (normalised by their number of occurrences in each clique)
    yOccurrences = sum(P.transpose() * P for P in P_sparse)   # Get the number of occurences of each y variable
    yOccurrencesDiag = np.array([yOccurrences[i, i] for i in range(yOccurrences.shape[0])])
    bAsArray = np.squeeze(np.array(b.todense()))       # Convert b sparse matrix into usable divisor
    bToSplit = np.divide(bAsArray, yOccurrencesDiag)   # Elementwise division

    for i in range(len(cliques)): 
        b_sparse[i] = np.matrix([bToSplit[cliques[i]]]).transpose()  # Populate b vectors (np.matrix)

    cliqueComponents = []
    for i in range(len(cliques)):
        cliqueComponents.append(CliqueComponent(At_sparse[i], b_sparse[i], c_sparse[i], K_sparse[i], P_sparse[i], options))

    return cliqueComponents


# Quick function to check input validity
def checkInputs(At, b, c, K):
    # Check each variable's type
    if type(At) != scipy.sparse.csc.csc_matrix or type(b) != scipy.sparse.csc.csc_matrix or type(c) != scipy.sparse.csc.csc_matrix:
        raise Exception("At, b and/or c is of incorrect type. Type should be scipy.sparse.csc.csc_matrix")

    # Check K structure fields
    if K.l is None or K.f is None or K.q is None or K.s is None:
        raise Exception("K structure does not have the required fields")

    # Modify K attribute if necessary
    if K.q.shape == (1, 1) and K.q[0, 0] == 0:
        K.q = scipy.array([0])

    # Convert attributes properly into integers
    K.f = K.f[0, 0]
    K.l = K.l[0, 0]
    K.s = K.s[0]

    # Add more checks later, if necessary
    return