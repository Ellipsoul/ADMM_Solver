from os.path import dirname, join as pjoin
import scipy.io as sio
import scipy
import matplotlib.pylab as plt

import numpy as np

from solver import admmCliqueSplitting
import time


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
t1 = time.process_time()
solution = admmCliqueSplitting(At, b, c, K)
t2 = time.process_time() - t1
print(t2)
# print("Update Y, time: ", solution.time.updateY)
# print("Update Z, time: ", solution.time.updateZ)
# print("Update X, time: ", solution.time.updateX)


print(solution.y)