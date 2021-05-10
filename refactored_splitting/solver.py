import scipy
import math
from scipy.sparse import csc_matrix, kron, vstack, csr_matrix, lil_matrix
from scipy.linalg import eigh
import numpy as np
from operator import itemgetter
import time

import networkx as nx
import matplotlib.pylab as plt

from helpers import ( vectoriseMatrix, matriciseVector, SolutionStructure, CliqueComponent, Options, CPUTime )


# Main ADMM Solver with Clique Splitting
def admmCliqueSplitting(At, b, c, K):
    t = time.process_time()  # Start the clock
    options = Options()      # Initialise default options

    # Initialise solution structure and set startup time
    sol = SolutionStructure(At, b, c, K, options)
    sol.time.start = t
    sol.time.setup = time.process_time() - sol.time.start

    printHeader()

    t1 = time.process_time()
    # Start of ADMM Iterations
    for i in range(options.maxIter):    
        # Print iteration if required
        # if i%options.dispIter==0 or i+1==options.maxIter:
        #     displayIteration(i, sol)

        # Check if stopping criterion has been satisfied
        if sol.iterationPrimalResidual < options.relTol and sol.iterationDualResidual < options.relTol: break

        # Carry out iteration steps
        updateYVector(sol)                          # Y vector minimisation step
        updateZProjection(sol)                      # z Conic Projection step
        updateSVector(sol)                          # s local vectors minimisation step
        updateLagrangeMultipliers(sol)              # Lagrange Multiplier update step
        gatherIterationResiduals(sol)               # Parse and calculate updated residuals

        sol.objectiveCost = -b.transpose() * sol.y  # Update objective function cost
    t2 = time.process_time() - t1
    print(t2)

    # Wrap up main function
    # displayIteration(i, sol)   # Final Iteration
    # print("|----------------------------------------------------------------|")
    # print("|  CPU time (s) = {:9.2e}".format(sol.time.elapsed))
    # print("| ADMM time (s) = {:9.2e}".format(sol.time.elapsed-sol.time.setup))
    # print("|    Iterations = {:6}".format(i))
    # print("|          Cost = {:6.4g}".format(sol.cost[0,0]))
    # print("|    Primal res = {:9.2e}".format(sol.pres))
    # print("|      Dual res = {:9.2e}".format(sol.dres))
    # print("|================================================================|")

    return sol


# Y vector minimisation step
def updateYVector(sol):
    for clique in sol.cliqueComponents:       # Iterating through each clique (can be parallelised)
        clique.updateYUpdateVector()          # Update righthand side of linear system
        clique.updateLMatrix()                # Update lefthand side of linear system (if required)

    # Sum righthand vector
    righthandVectorSum = sum(clique.yUpdateVector for clique in sol.cliqueComponents) + sol.options.lamb * sol.b

    sol.y = sol.Linv * righthandVectorSum  # Update y vector (Might be a better way to do this)


# z vector conic projection step
def updateZProjection(sol):
    for clique in sol.cliqueComponents:  # Iterate through all cliques
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
    # Iterate through all cliques (can be performed in parallel)
    for cl in sol.cliqueComponents:
        # Calculate column vector on righthand of equation
        rightHandSide = cl.rho * cl.P * sol.y + ( cl.sigma * cl.At.transpose() ) * (cl.c - cl.z + 1/cl.sigma * cl.eta) - cl.zeta + (1-cl.lamb) * cl.b

        oldS = cl.s                           # Temporarily store s vector of previous iteration 
        cl.s = cl.KKt.solve_A(rightHandSide)  # Update s vector by solving the prefactored cholesky matrix

        # Update first dual residual (Just locally first, needs to be summed across cliques to produce full residual)
        cl.dualResidualOne = cl.rho * cl.P.transpose() * (oldS - cl.s)
        # Update second residual (This is a local residual, and there will be one per clique)
        cl.dualResidualTwo = cl.sigma * cl.At * (cl.s - oldS)


# Lagrange multiplier update, computationally simple task (+ update local primal residual for each clique)
def updateLagrangeMultipliers(sol):
     for clique in sol.cliqueComponents:
        clique.eta += clique.sigma * (clique.c - clique.At*clique.s - clique.z)              # Update eta
        clique.zeta += clique.rho * (clique.s - clique.P * sol.y)                            # Update zeta

        # Update local primal residual
        clique.primalResidualLocal = np.linalg.norm(clique.c - clique.At * clique.s - clique.z)  


# Update the maximum primal and dual residual at this iteration
def gatherIterationResiduals(sol):
    cliques = sol.cliqueComponents

    primalResidualOne = max(c.primalResidualLocal for c in cliques)               # Max of local primal residuals
    primalResidualTwo = max(np.linalg.norm(c.s - c.P * sol.y) for c in cliques)   # Max of global primal residuals
    sol.iterationPrimalResidual = max(primalResidualOne, primalResidualTwo)

    dualResidualOne = np.linalg.norm(sum(c.dualResidualOne for c in cliques))     # Single first dual residual
    dualResidualTwoMax = max(np.linalg.norm(c.dualResidualTwo) for c in cliques)  # Max of all second dual residuals
    sol.iterationDualResidual = max(dualResidualOne, dualResidualTwoMax)

    # print("Primal", sol.iterationPrimalResidual)
    # print("Dual", sol.iterationDualResidual, '\n')


# Display current iteration information
def displayIteration(i, sol):
    sol.time.elapsed = time.process_time() - sol.time.start
    str = "|  {:4}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |  {:9.2e}  |"
    print(str.format(i, sol.objectiveCost, sol.iterationPrimalResidual, sol.iteraionDualResidual, sol.time.elapsed))
    

# Prints the header for solver
def printHeader():
    print("==================================================================")
    print("|                  ADMM WITH CLIQUE SPLITTING                    |")
    print("|================================================================|")
    print("|  ITER  |     COST    |    PRES     |    DRES     |    TIME     |")
    print("|--------|-------------|-------------|-------------|-------------|")