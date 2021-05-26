import scipy
import math
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, lil_matrix
from scipy.linalg import eigh
import numpy as np
from operator import itemgetter
import timeit
import csv

import networkx as nx
import matplotlib.pylab as plt

from joblib import Parallel, delayed

from helpers import ( vectoriseMatrix, matriciseVector, SolutionStructure, CliqueComponent, Options, CPUTime )


# Main ADMM Solver with Clique Splitting
def admmCliqueSplitting(At, b, c, K, numEl, omega):
    tStart = timeit.default_timer()  # Start the clock
    options = Options()              # Initialise default options

    # Initialise solution structure and set startup time
    sol = SolutionStructure(At, b, c, K, options)
    sol.data.numEl, sol.data.omega = numEl, omega
    sol.time.start = tStart
    sol.time.setupTime = timeit.default_timer() - sol.time.start

    printHeader()

    # Start of ADMM Iterations
    for i in range(options.maxIter):    
        # Print iteration if required, and store the data
        if (i%options.dispIter==0 or i==options.maxIter or i==1) and i != 0:
            displayIteration(i, sol)
            appendIterationData(i, sol)

        # Check if stopping criterion has been satisfied
        if sol.iterationPrimalResidual < options.relTol and sol.iterationDualResidual < options.relTol: break

        # Carry out iteration steps
        updateYVector(sol)                                  # Y vector minimisation step
        updateZProjection(sol)                              # z Conic Projection step
        updateSVector(sol)                                  # s local vectors minimisation step
        updateLagrangeMultipliers(sol)                      # Lagrange Multiplier update step
        gatherIterationResiduals(sol)                       # Parse and calculate updated residuals

        t0 = timeit.default_timer()
        sol.objectiveCost = (-sol.bt * sol.y)[0,0]   # Update objective function cost
        sol.time.calculateCost += timeit.default_timer() - t0  # Time the step

    # Wrap up main function
    displayIteration(i, sol)     # Final Iteration display
    appendIterationData(i, sol)  # Append data for final iteration
    appendFinalData(sol)         # Gather data to be exported as csv
    exportDataToCSV(sol.data)    # Export csv data
    print("|----------------------------------------------------------------|")
    print("|     CPU time (s) = {:9.2e}".format(sol.time.elapsed))
    print("|   Setup time (s) = {:9.2e}".format(sol.time.setupTime))
    print("| nxGraph time (s) = {:9.2e}".format(sol.time.findCliques))
    print("|    ADMM time (s) = {:9.2e}".format(sol.time.elapsed-sol.time.setupTime))
    print("|       Iterations = {:6}".format(i))
    print("|             Cost = {:6.4g}".format(sol.objectiveCost))
    print("|       Primal res = {:9.2e}".format(sol.iterationPrimalResidual))
    print("|         Dual res = {:9.2e}".format(sol.iterationDualResidual))
    print("|================================================================|")

    return sol

# Writing data to csv
def exportDataToCSV(data):
    # Iteration data into first file
    filepath1 = f'../results/splitting/splitting_iterations_{data.numEl}_{data.omega}.csv'
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
    filepath2 = f'../results/splitting/splitting_meta_{data.numEl}_{data.omega}.csv'
    with open(filepath2, mode='w') as csv_file2:
        fieldnames2 = ['num_el', 'omega', 'rel_tol', 'problem_numrows', 'problem_numcols', 'n_cliques', 
                        'avg_clique_numrows', 'avg_clique_numcols', 'total_time', 'setup_time', 'nx_time',
                        'admm_time', 'update_y_time', 'update_z_time', 'update_lagrange_time',
                        'update_residual_time', 'update_cost_time']
        writer = csv.DictWriter(csv_file2, fieldnames=fieldnames2)
        writer.writeheader()
        writer.writerow({'num_el': data.numEl, 'omega': data.omega, 'rel_tol': data.relTol, 
                            'problem_numrows': data.problemSize[0], 'problem_numcols': data.problemSize[1], 
                            'n_cliques': data.nCliques, 'avg_clique_numrows': data.avgCliqueSize[0], 
                            'avg_clique_numcols': data.avgCliqueSize[1], 'total_time': data.totalTime, 
                            'setup_time': data.setupTime, 'nx_time': data.nxTime, 'admm_time': data.admmTime, 
                            'update_y_time': data.updateYTime, 'update_z_time': data.updateZTime, 
                            'update_lagrange_time': data.updateLagrangeTime, 
                            'update_residual_time': data.updateResidualTime, 
                            'update_cost_time': data.updateCostTime})

# Y vector minimisation step
def updateYVector(sol):
    t0 = timeit.default_timer()               # Start the clock

    for clique in sol.cliqueComponents:       # Iterating through each clique (can be parallelised)
        clique.updateYUpdateVector()          # Update righthand side of linear system
        # clique.updateLMatrix()              # Update lefthand side of linear system (if required)

    # Sum righthand vector
    righthandVectorSum = sum(clique.yUpdateVector for clique in sol.cliqueComponents) + sol.options.lamb * sol.b

    sol.y = sol.Linv * righthandVectorSum  # Update y vector (Might be a better way to do this)

    sol.time.updateY += timeit.default_timer() - t0  # Time the step


# z vector conic projection step
def updateZProjection(sol):
    t0 = timeit.default_timer()  # Start the clock

    # Parallel(verbose=0, n_jobs=10)(delayed(updateZWrapper)(c) for c in sol.cliqueComponents)

    for clique in sol.cliqueComponents:  # Iterate through all cliques
        vectorToProject = clique.c - (clique.At * clique.s) + 1/clique.sigma * clique.eta  # Vector for conic projection
        clique.z = projectCones(vectorToProject, clique.K)                                 # Generate updated z vector 

    sol.time.updateZ += timeit.default_timer() - t0  # Time the step  

def updateZWrapper(clique):
    vectorToProject = clique.c - clique.At * clique.s + 1/clique.sigma * clique.eta  # Vector for conic projection
    clique.z = projectCones(vectorToProject, clique.K)                               # Generate updated z vector 

# Helper for conic projection of full vector
def projectCones(vector, K):
    projectZeroCone(vector[:K['f'], ])                     # Pass zero cone portion to helper
    projectNNOrthantCone(vector[K['f']:K['f']+K['l'], ])   # Pass NN Orthant portion to helper
    ptr = K['f']+K['l']
    for PSDSize in K['s']: 
        ptr2 = ptr + PSDSize ** 2
        vector[ptr:ptr2] = projectPSDCone(vector[ptr:ptr2])
        ptr = ptr2
    return vector


# Zero Cone Projection (Every large vector will have only one zero cone of some length)
def projectZeroCone(vector):
    vector[:, :] = 0


# Non-Negative Orthant Projection (Also only a single cone per clique, passed as a vector)
def projectNNOrthantCone(vector):
    vector[vector<0] = 0


# Positive Semidefinite Cone (There may be multiple of these cones within a clique)
def projectPSDCone(vector):
    M = matriciseVector(vector)     # Convert to matrix form
    M = 0.5 * (M + M.transpose())   # Ensure perfectly symmetric matrix

    E, U = eigh(M)                   # After processing, find eigenvalues and eigenvectors
    ind = [i for i in range(len(E)) if E[i] > 0]
    UsE = U[:, ind] * np.sqrt(E[ind])
    S = np.matmul(UsE, UsE.transpose())

    v = vectoriseMatrix(S).transpose()
    return v


# s vector update (currently using static inverse matrix)
def updateSVector(sol):
    t0 = timeit.default_timer()  # Start the clock

    # Iterate through all cliques (can be performed in parallel)
    for cl in sol.cliqueComponents:
        # Calculate column vector on righthand of equation
        # This long matrix operation is actually quite slow. Maybe there is a way to speed it up?
        rightHandSide = cl.rho * (csr_matrix(cl.P) * sol.y) + cl.sigma * (cl.A * (cl.c - cl.z + 1/cl.sigma * cl.eta)) - cl.zeta + (1-cl.lamb) * cl.b
        oldS = cl.s                           # Temporarily store s vector of previous iteration 
        cl.s = cl.KKt.solve_A(rightHandSide)  # Update s vector by solving the prefactored cholesky matrix

        # Update first dual residual (Just locally first, needs to be summed across cliques to produce full residual)
        cl.dualResidualOne = cl.rho * (cl.Pt * (oldS - cl.s))
        # Update second residual (This is a local residual, and there will be one per clique)
        cl.dualResidualTwo = cl.sigma * (cl.At * (cl.s - oldS))
    
    sol.time.updateS += timeit.default_timer() - t0  # Time the step


# Lagrange multiplier update, computationally simple task (+ update local primal residual for each clique)
def updateLagrangeMultipliers(sol):
    t0 = timeit.default_timer()  # Start the clock

    for clique in sol.cliqueComponents:
        clique.eta += clique.sigma * (clique.c - clique.At*clique.s - clique.z)              # Update eta
        clique.zeta += clique.rho * (clique.s - clique.P * sol.y)                            # Update zeta

        # Update local primal residual
        clique.primalResidualLocal = np.linalg.norm(clique.c - clique.At * clique.s - clique.z)
      
    sol.time.updateLagrangeMultipliers += timeit.default_timer() - t0  # Time the step


# Update the maximum primal and dual residual at this iteration
def gatherIterationResiduals(sol):
    t0 = timeit.default_timer()  # Start the clock

    cliques = sol.cliqueComponents

    primalResidualOne = max(c.primalResidualLocal for c in cliques)               # Max of local primal residuals
    primalResidualTwo = max(np.linalg.norm(c.s - c.P * sol.y) for c in cliques)   # Max of global primal residuals
    sol.iterationPrimalResidual = max(primalResidualOne, primalResidualTwo)

    dualResidualOne = np.linalg.norm(sum(c.dualResidualOne for c in cliques))     # Single first dual residual
    dualResidualTwoMax = max(np.linalg.norm(c.dualResidualTwo) for c in cliques)  # Max of all second dual residuals
    sol.iterationDualResidual = max(dualResidualOne, dualResidualTwoMax)

    sol.time.updateResiduals += timeit.default_timer() - t0  # Time the step


# Display current iteration information
def displayIteration(i, sol):
    sol.time.elapsed = timeit.default_timer() - sol.time.start
    str = "|  {:4}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |"
    cost = sol.objectiveCost if sol.objectiveCost else float('inf')
    print(str.format(i, cost, sol.iterationPrimalResidual, sol.iterationDualResidual, sol.time.elapsed))


# Append required iteration data
def appendIterationData(i, sol):
    sol.data.iteration.append(i)
    sol.data.objectiveCost.append(sol.objectiveCost)
    sol.data.primalResidual.append(sol.iterationPrimalResidual)
    sol.data.dualResidual.append(sol.iterationDualResidual)
    sol.data.time.append(sol.time.elapsed)


# Gathers overall ADMM data to be ready for exporting
def appendFinalData(sol):
    sol.data.totalTime = sol.time.elapsed
    sol.data.setupTime = sol.time.setupTime
    sol.data.nxTime = sol.time.findCliques
    sol.data.admmTime = sol.time.elapsed-sol.time.setupTime

    sol.data.updateYTime = sol.time.updateY
    sol.data.updateZTime = sol.time.updateZ
    sol.data.updateLagrangeTime = sol.time.updateLagrangeMultipliers
    sol.data.updateResidualTime = sol.time.updateResiduals
    sol.data.updateCostTime = sol.time.calculateCost


# Prints the header for solver
def printHeader():
    print("==================================================================")
    print("|                  ADMM WITH CLIQUE SPLITTING                    |")
    print("|================================================================|")
    print("|  ITER  |     COST    |    PRES     |    DRES     |    TIME     |")
    print("|--------|-------------|-------------|-------------|-------------|")