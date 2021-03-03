from os.path import dirname, join as pjoin
import scipy.io as sio

# Read data from file
mat_fname = './pop_data.mat'
mat_contents = sio.loadmat(mat_fname, struct_as_record=False)

# Data extraction
At = mat_contents['At']
b = mat_contents['b']
c = mat_contents['c']
K = mat_contents['K'][0,0]

