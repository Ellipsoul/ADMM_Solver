from os.path import dirname, join as pjoin
import scipy.io as sio
import scipy
import matplotlib.pylab as plt

import numpy as np

from solver import admmCliqueSplitting

# Read data from file
mat_fname = './pop_data.mat'
mat_contents = sio.loadmat(mat_fname, struct_as_record=False)

# Data extraction
# Solving problem of the form: min(-bt y) s.t. c - At y belonging to a conic set
At = mat_contents['At']
b = mat_contents['b']
c = mat_contents['c']        # c
K = mat_contents['K'][0,0]   # Class object with attributes f, l, q and s

# Call the solver
solution = admmCliqueSplitting(At, b, c, K)
# print("Update Y, time: ", solution.time.updateY)
# print("Update Z, time: ", solution.time.updateZ)
# print("Update X, time: ", solution.time.updateX)