import scipy
import math
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, lil_matrix
import numpy as np
from operator import itemgetter

import networkx as nx
import matplotlib.pylab as plt

#---------------------------------------------------------------------------------------------------------------------
# Detecting Cliques within Sparse A Matrix

def detectCliques(At, b, c, K):

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
    AtHead = At[:K.f+K.l, :]   # Matrix of equality + inequality constraints to be stacked with collapsed PSDs
    collapsedRows = []
    
    currentIdx = K.f + K.l
    # Iterate through all PSD constraints
    ptr = K.f + K.l
    for matrixSize in K.s:
        rowsToExtract = matrixSize ** 2
        psdConstraint = AtOnes[currentIdx: currentIdx+rowsToExtract, :]         # Retrive the matrix subset
        cConstraint = c[currentIdx: currentIdx+rowsToExtract]                   # Retrive c vector subset

        collapsedRow = psdConstraint.sum(axis=0)                                # Collapse into a single vector
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

    for i in range(len(cliques)): b_sparse[i] = bToSplit[cliques[i]]  # Populate b vectors

    return (At_sparse, b_sparse, c_sparse, K_sparse, P_sparse, len(cliques))


# Y vector minimisation step
def updateYVector(cliqueComponents, y, b, options):
    for clique in cliqueComponents:           # Iterating through each clique (can be parallelised)
        clique.updateYUpdateVector()          # Update righthand side of linear system
        clique.updateLMatrix()                # Update lefthand side of linear system (if required)

     # Sum righthand vector
    righthandVectorSum = sum(clique.yUpdateVector for clique in cliqueComponents) + options['lamb'] * b

    # Gather lefthand matarix inverse
    lefthandMatrixSum = sum(clique.L for clique in cliqueComponents)              # Sum lefthand diagonal matrix                 
    diagReciprocals = np.reciprocal(lefthandMatrixSum.data)     # Gather reciprocals of diagonal
    nDiag = len(diagReciprocals)
    Linv = scipy.sparse.dia_matrix((diagReciprocals, [0]), shape=(nDiag, nDiag))  # Create matrix inverse

    y = Linv * righthandVectorSum  # Update y vector (Might be a better way to do this)


# Z vector conic projection step
def updateZProjection(cliqueComponents):
    for clique in cliqueComponents:  # Iterate through all cliques
        vectorToProject = clique.c - clique.At * clique.s - 1/clique.sigma * clique.eta  # Vector for conic projection

        # TODO: What is the NN orthant projections for? Seems like only cones or single values
        print(vectorToProject)
        print(clique.K)
        print('')


#