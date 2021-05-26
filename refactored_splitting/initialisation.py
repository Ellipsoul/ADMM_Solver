from os.path import dirname, join as pjoin
import scipy.io as sio
import scipy
import matplotlib.pylab as plt

import numpy as np

from solver import admmCliqueSplitting
import time

# POP Problems will be run from:
# omega = 3, 3-4
# omega = 2, 3-15
# omega = 1, 3-101

# Read data from file
numEl = 3
numElStr = str(numEl)
omega = 1
omegaStr = str(omega)
pop_problem = f'_{numElStr}_{omegaStr}.mat'
mat_fname = f'./../popData/pop_data{pop_problem}'

mat_contents = sio.loadmat(mat_fname, struct_as_record=False)

# Data extraction
# Solving problem of the form: min(-bt y) s.t. c - At y belonging to a conic set
At = mat_contents['At']
b = mat_contents['b']
c = mat_contents['c']        # c
K = mat_contents['K'][0,0]   # Class object with attributes f, l, q and s

# Call the solver
solution = admmCliqueSplitting(At, b, c, K, numEl, omega)
print("|  Update Global Y Vector,      time: ", solution.time.updateY)
print("|  Update Z Projection,         time: ", solution.time.updateZ)
print("|  Update Local S Vectors,      time: ", solution.time.updateS)
print("|  Update Lagrange Multipliers, time: ", solution.time.updateLagrangeMultipliers)
print("|  Update Residual,             time: ", solution.time.updateResiduals)
print("|  Calculate Objective Cost,    time: ", solution.time.calculateCost)
print("|  Number of Cliques in problem:      ", solution.ncliques)
print("|================================================================|")