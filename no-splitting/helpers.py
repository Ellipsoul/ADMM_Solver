import scipy
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, dia_matrix, identity, csc_matrix
from scipy.sparse.linalg import inv
import math
import numpy as np
import matplotlib.pylab as plt
from sksparse.cholmod import cholesky_AAt

################################################################################
# Class to store solution variables
class solStructure:
    def __init__(self, At, b, c, K, options):
        self.A = At.transpose()
        self.bt = b.transpose()

        # x: lagrange multipliers
        self.x = np.zeros(shape=(c.shape[0], 1))

        # y: free variables
        self.y = np.ones(shape=(b.shape[0], 1))

        # z: conic variables
        zeroCones = np.zeros(shape=(K.f, 1))                              # Equality: 0s
        nnOrthants = np.zeros(shape=(K.l, 1))                              # Inequality: 1s
        PSDs = [np.identity(size).reshape((size**2, 1)) for size in K.s]  # PSD: identity
        self.z = vstack([zeroCones, nnOrthants, *PSDs])                      # Stack up the vectors

        # Primal and dual residuals
        self.pres = np.linalg.norm(c - At*self.y - self.z)
        self.dres = float('inf')

        # Cost
        self.cost = self.bt * self.y

        # Time
        self.time = CPUTime()

        # Factorization of system matrix
        self.KKT = cholesky_AAt( csc_matrix(self.A), 0.0 )


################################################################################
# Class to store CPU times
class CPUTime:
    def __init__(self, start=0):
        self.start = start
        self.init = 0.0
        self.elapsed = 0.0
        self.updateX = 0.0
        self.updateY = 0.0
        self.updateZ = 0.0


################################################################################
# Class of options
class Options:
    def __init__(self, rho=10, relTol=1.0e-06, maxIter=10000, dispIter=100):
        self.rho = rho                      # Penalty parameter
        self.relTol = relTol                # Relative tolerance
        self.maxIter = maxIter              # Maximum number of iterations
        self.dispIter = dispIter            # Iterations for which to print

################################################################################
# Helper function to vectorise a matrix, taking in a sparse matrix
def vectoriseMatrix(M):
    n = M.shape[0]**2               # Length of long vector
    return M.reshape((n, 1))        # Reshape matrix in place with no return value


################################################################################
# Helper function for converting a vector back into a square matrix
def matriciseVector(v):
    n = int(math.sqrt(v.shape[0]))
    return v.reshape((n, n))


################################################################################
# Function to check input validity
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
