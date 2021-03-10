from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib.pylab as plt

from solverhelpers import checkInputs, splitBlocks, detectCliques

# Read data from file
mat_fname = './pop_data.mat'
mat_contents = sio.loadmat(mat_fname, struct_as_record=False)

# Data extraction
# Solving problem of the form: min(bt y) s.t. c - At y belonging to a conic set
At = mat_contents['At']
b = mat_contents['b']
c = mat_contents['c']        # c
K = mat_contents['K'][0,0]   # Class object with attributes f, l, q and s

# Changeable options for the solver
options = {
    "solver": 'hsde',           # Solver type (keep just one for now)
    "relTol": 1.0000e-04,       #
    "rescale": 1,               #
    "verbose": 1,               # Additional comments during solver running
    "dispIter": 50,             #
    "maxIter": 1000,            # Maximum number of iteration
    "chordalize": 1,            #
    "yPenalty": 1,              # 
    "completion": 1,            # 
    "rho": 1,                   # Rho penalty parameter
    "adaptive": 1,              #
    "mu": 2,                    # 
    "nu": 10,                   # 
    "rhoMax": 1000000,          #
    "rhoMin": 1.0000e-06,       #
    "rhoIt": 10,                #
    "KKTfact": 'blk',           #
    "alpha": 1.8000,            #
}

checkInputs(At, b, c, K)           # Verify inputs
# Split semidefinite blocks into distinct blocks (NOT NECESSARY FOR NOW)
# (At, b, c, K, options) = splitBlocks(At, b, c, K, options) 
# options['n'], options['m'] = At.shape

# Detect cliques, dividing up problem into separate sections
(At_sparse, b_sparse, c_sparse) = detectCliques(At, b, c, K)
