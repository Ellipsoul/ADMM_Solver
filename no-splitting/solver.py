import scipy
import math
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, lil_matrix
from scipy.linalg import eigh
import numpy as np
from operator import itemgetter
import matplotlib.pylab as plt
from helpers import vectoriseMatrix, matriciseVector, checkInputs
from helpers import solStructure, Options, CPUTime
import timeit
import csv
#import sksparse.cholmod

###############################################################################
# Update Y (solve system)
def updateY(sol, At, b, c, K, options):
    t0 = timeit.default_timer()
    rhs = sol.A * ( c - sol.z - sol.x/options.rho  ) + b/options.rho
    # sol.y = scipy.sparse.linalg.spsolve(H,rhs) # this is extremely ugly: should factorize beforehand!
    sol.y = sol.KKT.solve_A(rhs)
    sol.y = sol.y.reshape(sol.y.shape[0],1) # this is very ugly!
    sol.time.updateY += timeit.default_timer() - t0


###############################################################################
# Update z: project on cones
# Also update the dual residual, since we need the old and new z variables for this
# We do not time the residual update
def updateZ(sol, At, b, c, K, options):
    t0 = timeit.default_timer()
    vectorToProject = c - At*sol.y - sol.x/options.rho
    zNew = projectCones(vectorToProject, K)
    sol.time.updateZ += timeit.default_timer() - t0
    sol.dres = np.linalg.norm( sol.A * (sol.z-zNew) ) * options.rho
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
    t0 = timeit.default_timer()
    pres = At*sol.y + sol.z - c
    sol.x += options.rho * pres
    sol.time.updateX += timeit.default_timer() - t0
    sol.pres = np.linalg.norm(pres)



###############################################################################
# Display current iteration
def displayIteration(i, sol):
    sol.time.elapsed = timeit.default_timer() - sol.time.start
    str = "|  {:4}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |"
    print(str.format(i, sol.cost[0,0], sol.pres, sol.dres, sol.time.elapsed))


###############################################################################
# Main solver
def admmSolverNoSplitting(At, b, c, K, numEl, omega):
    """
    Solve conic problem in standard dual form using ADMM.

    The problem takes the form

        min   -b'*y
        s.t.   c - At*y ∈ K

    where y \in R^m and K is a product of the zero cone, the nonnegative orthant,
    and spositive semidefinite cones.
    """

    # Start the clock
    t = timeit.default_timer()

    # Verify inputs
    checkInputs(At, b, c, K)
    options = Options()

    # Initialise solution structure and set startup time
    sol = solStructure(At, b, c, K, options, numEl, omega)
    sol.time.start = t
    sol.time.setup = timeit.default_timer() - sol.time.start

    # Print header
    print("==================================================================")
    print("|                      ADMM WITHOUT SPLITTING                    |")
    print("|================================================================|")
    print("|  ITER  |     COST    |    PRES     |    DRES     |    TIME     |")
    print("|--------|-------------|-------------|-------------|-------------|")

    # ADMM Iterations
    for i in range(options.maxIter):

        # Print
        if (i%options.dispIter==0 or i+1==options.maxIter or i==1) and i!=0:
            displayIteration(i, sol)
            appendIterationData(i, sol)

        # Are we done?
        # NOTE: Residuals have been updated by updateZ and updateX
        # updateResiduals(sol, At, b, c, K, options)
        if sol.pres < options.relTol and sol.dres < options.relTol:
            break

        # Update
        updateY(sol, At, b, c, K, options)
        updateZ(sol, At, b, c, K, options)
        updateX(sol, At, b, c, K, options)
        sol.cost = -sol.bt * sol.y
        


    # Terminate main function
    displayIteration(i, sol)
    appendFinalData(sol)
    exportDataToCSV(sol.data)
    print("|----------------------------------------------------------------|")
    print("|  CPU time (s) = {:9.2e}".format(sol.time.elapsed))
    print("| ADMM time (s) = {:9.2e}".format(sol.time.elapsed-sol.time.setup))
    print("|    Iterations = {:6}".format(i))
    print("|          Cost = {:6.4g}".format(sol.cost[0,0]))
    print("|    Primal res = {:9.2e}".format(sol.pres))
    print("|      Dual res = {:9.2e}".format(sol.dres))
    print("|================================================================|")
    return sol

# Append required iteration data
def appendIterationData(i, sol):
    sol.data.iteration.append(i)
    sol.data.objectiveCost.append(sol.cost[0, 0])
    sol.data.primalResidual.append(sol.pres)
    sol.data.dualResidual.append(sol.dres)
    sol.data.time.append(sol.time.elapsed)


# Gathers overall ADMM data to be ready for exporting
def appendFinalData(sol):
    sol.data.totalTime = sol.time.elapsed
    sol.data.setupTime = sol.time.setup
    sol.data.admmTime = sol.time.elapsed-sol.time.setup

    sol.data.updateYTime = sol.time.updateY
    sol.data.updateZTime = sol.time.updateZ
    sol.data.updateLagrangeTime = sol.time.updateX

# Writing data to csv
def exportDataToCSV(data):
    # Iteration data into first file
    filepath1 = f'../results/no_splitting/nosplitting_iterations_{data.numEl}_{data.omega}.csv'
    with open(filepath1, mode='w') as csv_file1:
        fieldnames1 = ['i', 'objective_cost', 'primal_residual', 'dual_residual', 'time']
        writer = csv.DictWriter(csv_file1, fieldnames=fieldnames1)

        writer.writeheader()
        for i in range(len(data.iteration)):
            writer.writerow({'i': data.iteration[i], 
                            'objective_cost': data.objectiveCost[i], 
                            'primal_residual': data.primalResidual[i],
                            'dual_residual': data.dualResidual[i],
                            'time': data.time[i]})
    
    # Meta data into another file
    filepath2 = f'../results/no_splitting/nosplitting_meta_{data.numEl}_{data.omega}.csv'
    with open(filepath2, mode='w') as csv_file2:
        fieldnames2 = ['num_el', 'omega', 'rel_tol', 'problem_numrows', 'problem_numcols', 'total_time', 
                        'setup_time', 'admm_time', 'update_y_time', 'update_z_time', 'update_lagrange_time']
        writer = csv.DictWriter(csv_file2, fieldnames=fieldnames2)
        writer.writeheader()
        writer.writerow({'num_el': data.numEl, 'omega': data.omega, 'rel_tol': data.relTol, 
                            'problem_numrows': data.problemSize[0], 'problem_numcols': data.problemSize[1], 
                            'total_time': data.totalTime, 'setup_time': data.setupTime, 'admm_time': data.admmTime, 
                            'update_y_time': data.updateYTime, 'update_z_time': data.updateZTime, 
                            'update_lagrange_time': data.updateLagrangeTime})