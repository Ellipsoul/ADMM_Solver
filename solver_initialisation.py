from os.path import dirname, join as pjoin
import scipy.io as sio
import scipy
import matplotlib.pylab as plt

import numpy as np

from solver_helpers import checkInputs, splitBlocks, CliqueComponent
from solver_functions import ( detectCliques, updateYVector, updateZProjection, updateSVector, 
    updateLagrangeMultipliers)

# Read data from file
mat_fname = './pop_data.mat'
mat_contents = sio.loadmat(mat_fname, struct_as_record=False)

# Data extraction
# Solving problem of the form: min(bt y) s.t. c - At y belonging to a conic set
At = mat_contents['At']
b = mat_contents['b']
c = mat_contents['c']        # c
K = mat_contents['K'][0,0]   # Class object with attributes f, l, q and s

checkInputs(At, b, c, K)           # Verify inputs

# Split semidefinite blocks into distinct blocks (NOT NECESSARY FOR NOW)
# (At, b, c, K, options) = splitBlocks(At, b, c, K, options) 
# options['n'], options['m'] = At.shape

# Detect cliques, decomposing problem into separate parallel parts
# Each variable will be a list of options
(At_sparse, b_sparse, c_sparse, K_sparse, P_sparse, numCliques) = detectCliques(At, b, c, K)

# Definable options
options = {
    "maxIter": 1000,               # Maximum number of iterations
    "relTol": 1.0000e-04,       # Relative tolerance parameter for 
    "rho": 10,                  # Penalty parameter for objective function
    "sigma": 10,                # Penalty parameter for constraints
    "eta0": 1,                  # Initial eta vector values
    "zeta0": 1,                 # Initiali zeta vector values
    "lamb": 0.5                 # Default relative weighting for cost
}

# Initialise list of CliqueComponent classes for each clique
cliqueComponents = [None for _ in range(numCliques)]
for i in range(numCliques):
    cliqueComponents[i] = CliqueComponent(At_sparse[i], b_sparse[i], c_sparse[i], K_sparse[i], P_sparse[i], options)
y = np.ones(shape=(b.shape[0], 1))   # Initialise y vector

# Main ADMM Solver Iteration
#-------------------------------------------------------------------------------------------------------------------

# Run algorithm
for i in range(options['maxIter']):
    y = updateYVector(cliqueComponents, y, b, options)          # Update y vector
    print(y)
    print('')
    updateZProjection(cliqueComponents, options)                # Update z vector conic projection
    updateSVector(cliqueComponents, y)                          # Update s vector
    updateLagrangeMultipliers(cliqueComponents, y)              # Update Lagrange multipliers