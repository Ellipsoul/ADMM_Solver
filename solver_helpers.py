import scipy
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix
from scipy.sparse.linalg import inv

import numpy as np
import matplotlib.pylab as plt

# Class to wrap all variables within each subproblem
class CliqueComponent:
    def __init__(self, At, b, c, K, P, options):
        # Problem statement components
        self.At = At
        self.b = b
        self.c = c
        self.K = K
        self.P = P
        
        # Useful constants
        self.rho = options["rho"]
        self.sigma = options["sigma"]
        self.lamb = options["lamb"]

        # Lagrange multiplier vectors to be updated at each cycle
        self.zeta = np.ones(shape=(self.b.shape[0], 1))
        self.eta = np.ones(shape=(self.At.shape[1], 1))

        # Initialise local s vector containing extracted components on the full y vector
        self.s = np.zeros(shape=(self.b.shape[0], 1))
        
        # Initialise local z cost vector for ensuring problem remains conic
        zeroCones = np.zeros(shape=(K["f"], 1))                              # Equality: 0s
        nnOrthants = np.ones(shape=(K["f"], 1))                              # Inequality: 1s
        PSDs = [np.identity(size).reshape((size**2, 1)) for size in K["s"]]  # PSD: identity
        self.z = vstack([zeroCones, nnOrthants, *PSDs])                      # Stack up the vectors for constraints

        self.L = self.rho * self.P.transpose() * self.P                  # Generate static matrix: L = rho_i*Pt*P

        # self.yUpdateVector = self.P.transpose() * (self.zeta + self.rho * self.s)    # Initialise first y updating value

    # Generate righthand local value for minimising with respect ot y
    def updateYVector(self):
        self.yUpdateVector = self.P.transpose() * (self.zeta + self.rho * self.s)


# Helper function to vectorise a matrix, taking in a sparse matrix
def vectoriseMatrix(M):
    n = M.shape[0]**2               # Length of long vector
    M.reshape((1, n), copy=True)    # Reshape matrix in place with no return value
    return M

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
    print("All checks passed!\n")
    return

#---------------------------------------------------------------------------------------------------------------------
# Section ensures matrices are sparsified and splits blocks appropriately

# Attempts to split semidefinite blocks into smaller connected components
def splitBlocks(At, b, c, K, options):
    nVectorConstraints = K.f + K.l + sum(K.q)    # Number of vector constaints
    nMatrixConstraints = len(K.s)                # Number of semidefinite cone constraints

    # Retrive sparsity pattern of semidefinite blocks
    cSemidefinitePattern = c[nVectorConstraints:]
    AtSemidefinitePattern = At[nVectorConstraints:, :]  # Removing rows in c and At equal to number of vector constraints
    SPfromC = cSemidefinitePattern.copy().tocsr()
    SPfromC.data.fill(1)
    
    SPfromAt = AtSemidefinitePattern.copy().tocsr()
    SPfromAt.data.fill(1)
    
    SP = (SPfromC + csc_matrix(SPfromAt.sum(axis=1))).tocsr()
    SP.data.fill(1)
    
    # Initialising necessary variables
    R, rowCount = [], 0
    C, colCount = [], 0
    dimensions = []

    # Looping over blocks to find connected components:
    for i in range(nMatrixConstraints):
        m = K.s[i]
        B = csc_matrix(SP[rowCount:rowCount+m**2].reshape((m, m), order='C', copy=True))
        (tags, nComponents) = findConnectedComponents(B)   # Calling helper to retrieve number of components

        if nComponents > 1:  # Split if more than one component found
            # Determine component dimensions, permutation matrix and block pattern indices
            pattern = np.zeros((m,), dtype=int)
            count = 0
            I, J = np.array([], dtype=int), np.array([], dtype=int)

            # Iterate through all the components
            for j in range(1, nComponents + 1):
                # Dimensions and permutation vector
                componentIdxs = np.where(tags == j)[0]  # Extract indices of a found component
                dimension = componentIdxs.size
                dimensions.append(dimension)            # Store dimensions in an array
                linearIdxs = np.array([i for i in range(count, count+dimension)])
                pattern[linearIdxs] = componentIdxs     # Insert the component indices into the long pattern array

                # Find indices for block pattern
                rows, cols = np.meshgrid(linearIdxs, linearIdxs)
                rows = rows.flatten('F')
                cols = cols.flatten('F')
                I = np.append(I, rows)
                J = np.append(J, cols)

                count += dimension  # Update counter by dimension

            # Project onto connected components
            P = csc_matrix(([1 for _ in range(m)], (pattern, [i for i in range(m)])), shape=(m, m))  # Permuation of cone
            B = csc_matrix(([1 for _ in range(len(I))], (I, J)), shape=(m, m))  # Permuted connected components

            entryIndices = np.argwhere(B)
            entryIndicesFlattened = [(i[1] + i[0]*m) for i in entryIndices]
            numberOfEntries = len(entryIndicesFlattened)
            # Projection onto component block
            E = csc_matrix(([1 for _ in range(numberOfEntries)], (entryIndicesFlattened, [i for i in range(numberOfEntries)])), shape=(m**2, numberOfEntries))

            # Calculate and store indices
            kronekerMatrix = kron(P, P) * E
            kronekerIndices = np.argwhere(kronekerMatrix)
            Ri, Ci = [i[0]+rowCount for i in kronekerIndices], [i[1]+colCount for i in kronekerIndices]
            R.extend(Ri)
            C.extend(Ci)

            # Update row and column counters
            rowCount += m**2
            colCount += numberOfEntries

        else:  # No components, just update some variables
            dimensions.append(m)
            RiRow = [i+rowCount for i in range(m**2)]
            RiCol = [i+colCount for i in range(m**2)]
            R.extend(RiRow)
            C.extend(RiCol)
            rowCount += m**2
            colCount += m**2

    # Finally, aggregate and update problem data
    M = csc_matrix(([1 for _ in range(len(R))], (R, C)), shape=(rowCount, colCount)).transpose()
    At = vstack([At[:nVectorConstraints, :], M*AtSemidefinitePattern])
    c = vstack([c[:nVectorConstraints], M*cSemidefinitePattern])
    K.s = dimensions

    # Retrieve the used variables (sorted but maybe shouldn't be?)
    options['usedvars'] = np.concatenate(([i for i in range(nVectorConstraints)], R+nVectorConstraints))
    return  (At, b, c, K, options) # Return modified variables to main caller


# Helper for splitBlocks to find connected components for the sparsity pattern matrix B
# Uses a breadth first search technique
def findConnectedComponents(B):
    L = B.shape[0]  # Number of vertices
    tags = np.zeros((L,), dtype=int)        # Initialise array of tags
    # rts = []                               # Tracking order in which variables are explored (not in use)
    nConnectedComponents = 0                 # Track number of connected components
    
    unexploredIdxs = np.where(tags==0)[0]    # Find indices that are still empty

    # Continue until all tags have been filled (all vertices explored)
    while unexploredIdxs.size != 0:
        firstUnexploredVertex = unexploredIdxs[0]           # Explore vertices one at a time
        # rts.append(firstUnexploredVertex) 
        lst = np.array([firstUnexploredVertex])             # Initialise queue with initial unexplored vertex
        nConnectedComponents += 1                           # Increment connected components found
        tags[firstUnexploredVertex] = nConnectedComponents  # Indicate number of connected components at this index

        # Breadth first search to clear the queue
        while True:
            newList = []                                    # Initialise new list
            for lc in range(len(lst)):                      # Going through points to explore
                p = lst[lc]                                 # Determine the number of connected components and update
                cp = np.argwhere(B[p, :])
                cp1 = cp[tags[cp] == 0]                     # Vertices to add to the queue
                
                tags[cp1] = nConnectedComponents
                newList = np.concatenate((newList, cp1))    # Add the vertices to explore to the queue
            lst = newList

            if not lst.any(): break                         # Escape if queue has been emptied
        
        unexploredIdxs = np.where(tags==0)[0]               # Update unexplored indices
    
    nComponents = np.max(tags)                                 # Report number of components and return
    return (tags, nComponents)