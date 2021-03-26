import scipy
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix
import numpy as np

import networkx as nx
import matplotlib.pylab as plt

#---------------------------------------------------------------------------------------------------------------------
# Detecting Cliques within Sparse A Matrix

def detectCliques(At, b, c, K):

    # Replace all non-zero data with just ones
    AtOnes = At.copy().tocsr()
    AtOnes.data.fill(1)

    # K.f - Fixed, K.l - Linear, K.s[] - Semidefinite
    # Collapse single matrix constrants into single row
    AtHead = At[:K.f+K.l, :]   # Initialise new matrix with original equality and inequality constraints
    collapsedRows = []

    currentIdx = K.f + K.l
    # Iterate through all PSD constraints
    for matrixSize in K.s:
        rowsToExtract = matrixSize ** 2
        psdConstraint = AtOnes[currentIdx: currentIdx+rowsToExtract, :]   # Retrive the matrix subset
        collapsedRow = psdConstraint.sum(axis=0)                          # Collapse into a single vector
        collapsedRows.append(collapsedRow)                                # Store the collapsed row

        currentIdx += rowsToExtract                                       # Increment row counter

    AtCollapsed = vstack([AtHead] + collapsedRows)                        # Stack rows together

    # Find cliques
    S = AtCollapsed.transpose() * AtCollapsed                             # Generate matrix of codependencies
    G = nx.Graph(S)                                                       # Initialise NetworkX graph
    cliques = list(nx.algorithms.find_cliques(G))                         # Retrieve cliques

    # Gather the related
    print(cliques)

    plt.spy(AtCollapsed)
    plt.show()

    return (1, 1, 1)