import scipy
import math
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, lil_matrix
from scipy.linalg import eigh
import numpy as np
from operator import itemgetter
import matplotlib.pylab as plt
from helpers import vectoriseMatrix, matriciseVector, checkInputs
from helpers import solStructure, Options, CPUTime
import time
#import sksparse.cholmod

###############################################################################
# Update Y (solve system)
def updateY(sol, At, b, c, K, options):
    t0 = time.process_time()
    rhs = At.transpose() * ( c - sol.z - sol.x/options.rho  ) + b/options.rho
    # sol.y = scipy.sparse.linalg.spsolve(H,rhs) # this is extremely ugly: should factorize beforehand!
    sol.y = sol.KKT.solve_A(rhs)
    sol.y = sol.y.reshape(sol.y.shape[0],1) # this is very ugly!
    sol.time.updateY += time.process_time() - t0


###############################################################################
# Update z: project on cones
# Also update the dual residual, since we need the old and new z variables for this
# We do not time the residual update
def updateZ(sol, At, b, c, K, options):
    t0 = time.process_time()
    vectorToProject = c - At*sol.y - sol.x/options.rho
    zNew = projectCones(vectorToProject, K)
    sol.time.updateZ += time.process_time() - t0
    sol.dres = np.linalg.norm( At.transpose() * (sol.z-zNew) ) * options.rho
    sol.z = zNew


# Helper for conic projection of full vector
def projectCones(vector, K):
    projectZeroCone(vector[:K.f, 0])                     # Pass zero cone portion to helper
    projectNNOrthantCone(vector[K.f:K.f+K.l, 0])   # Pass NN Orthant portion to helper
    ptr = K.f+K.l
    for PSDSize in K.s:
        ptr2 = ptr + PSDSize ** 2
        vector[ptr:ptr2] = projectPSDCone(vector[ptr:ptr2])
        ptr = ptr2
    return vector

# Zero Cone Projection (Every large vector will have only one zero cone of some length)
def projectZeroCone(vector):
    vector[:, :] = 0

# Non-Negative Orthant Projection (Also only a single cone per clique, passed as a vector)
def projectNNOrthantCone(vector):
    # ind = [i for i in range(len(vector)) if vector[i] < 0]
    vector[vector<0] = 0

# Positive Semidefinite Cone (There may be multiple of these cones within a clique)
def projectPSDCone(vector):
    M = matriciseVector(vector)
    M = 0.5 * (M + M.transpose())
    E, U = eigh(M)                   # After processing, find eigenvalues and eigenvectors
    ind = [i for i in range(len(E)) if E[i] > 0]
    UsE = U[:, ind] * np.sqrt(E[ind])
    S = np.matmul(UsE, UsE.transpose())
    v = vectoriseMatrix(S)
    return v


###############################################################################
# Lagrange multiplier update, computationally simple task
# Also update primal residual to minimize number of operations
# We do not time the residual update
def updateX(sol, At, b, c, K, options):
    t0 = time.process_time()
    pres = At*sol.y + sol.z - c
    sol.x += options.rho * pres
    sol.time.updateX += time.process_time() - t0
    sol.pres = np.linalg.norm(pres)



###############################################################################
# Display current iteration
def displayIteration(i, sol):
    sol.time.elapsed = time.process_time() - sol.time.start
    str = "|  {:4}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |"
    print(str.format(i, sol.cost[0,0], sol.pres, sol.dres, sol.time.elapsed))


###############################################################################
# Main solver
def admmSolverNoSplitting(At, b, c, K):
    """
    Solve conic problem in standard dual form using ADMM.

    The problem takes the form

        min   -b'*y
        s.t.   c - At*y ∈ K

    where y \in R^m and K is a product of the zero cone, the nonnegative orthant,
    and spositive semidefinite cones.
    """

    # Start the clock
    t = time.process_time()

    # Verify inputs
    checkInputs(At, b, c, K)
    options = Options()

    # Initialise solution structure and set startup time
    sol = solStructure(At, b, c, K, options)
    sol.time.start = t
    sol.time.setup = time.process_time() - sol.time.start

    # Print header
    print("==================================================================")
    print("|                      ADMM WITHOUT SPLITTING                    |")
    print("|================================================================|")
    print("|  ITER  |     COST    |    PRES     |    DRES     |    TIME     |")
    print("|--------|-------------|-------------|-------------|-------------|")

    # ADMM Iterations
    for i in range(options.maxIter):

        # Print
        if i%options.dispIter==0 or i+1==options.maxIter:
            displayIteration(i, sol)

        # Are we done?
        # NOTE: Residuals have been updated by updateZ and updateX
        # updateResiduals(sol, At, b, c, K, options)
        if sol.pres < options.relTol and sol.dres < options.relTol:
            break

        # Update
        updateY(sol, At, b, c, K, options)
        updateZ(sol, At, b, c, K, options)
        updateX(sol, At, b, c, K, options)
        sol.cost = -b.transpose() * sol.y


    # Terminate main function
    displayIteration(i, sol)
    print("|----------------------------------------------------------------|")
    print("|  CPU time (s) = {:9.2e}".format(sol.time.elapsed))
    print("| ADMM time (s) = {:9.2e}".format(sol.time.elapsed-sol.time.setup))
    print("|    Iterations = {:6}".format(i))
    print("|          Cost = {:6.4g}".format(sol.cost[0,0]))
    print("|    Primal res = {:9.2e}".format(sol.pres))
    print("|      Dual res = {:9.2e}".format(sol.dres))
    print("|================================================================|")
    return sol
