import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import timeit

# Loading the pickled data back
with open('allData.pk', 'rb') as fi:
    noSplittingIterationData, noSplittingMetaData, splittingIterationData, splittingMetaData = pickle.load(fi)

"""
Plots to make

- Setup time vs numEl
Splitting and nosplitting, maybe plot on separate graphs
Explain that setup is negligible for nosplitting since cliques don't need to be detected
Also a lot more overhead with moving the cliques to the 
Observe the additional time required for clique detection as it increases

- nxTime vs numEl
Splitting only. Might be useful to see exactly how much time this part takes
Maybe plot with the setup time to see how much of the setup time is nx Time

- Update y Time vs numEl
Splitting and no splitting. These are expect to be very similar, so this is an important check

- Update z Time vs numEl
Splitting and no splitting. This projection time will be different since splitting has to do it for all cliques
Need to comment on the difference, and mention that with a for loop/non-parallelised it's definitely slower

- Update Lagrange time
Splitting and no splitting, same as above

(Not going to plot update residual and update objective function time)

- Breakdown of time taken for each step (pie chart)
Splitting and no splitting. Really important to show which steps are taking the most time
Have an 'other' segment for anything that wasn't timed
Show a few pie charts for different problem sizes to observe the differences
"""

#-------------------------------------------------------------------------------------------------------------------
"""
- Problem size vs Number of Elements
Might be useful as a start to show how big the A matrices get as number of elements increase
Plot for rows and columns on the same graph
Also shows how quickly increasing omega increases the problem size 
"""

"""
# Plot for omega = 1
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df = noSplittingMetaData.loc[noSplittingMetaData["omega"]==1]
plt.plot(df['num_el_real'], df['problem_numrows'], linewidth=3)
plt.plot(df['num_el_real'], df['problem_numcols'], linewidth=3)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Rows and Columns vs Number of Independent Variables", fontproperties=titleDict)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$", fontdict=subtitleDict, pad=14)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("# Rows/Columns", fontsize=10)
plt.legend(["# Rows", "# Columns"], loc=0)

# Limits and interval ticks
plt.xlim(0, 225)
plt.ylim(0, 6500)
plt.xticks(np.arange(0, 226, 25))
plt.yticks(np.arange(0, 6501, 500))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/rowscols_vs_numel_omega1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 2
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df = noSplittingMetaData.loc[noSplittingMetaData["omega"]==2]
plt.plot(df['num_el_real'], df['problem_numrows'], linewidth=3)
plt.plot(df['num_el_real'], df['problem_numcols'], linewidth=3)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Rows and Columns vs Number of Independent Variables", fontproperties=titleDict)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=14)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("# Rows/Columns", fontsize=10)
plt.legend(["# Rows", "# Columns"], loc=0)

# Limits and interval ticks
plt.xlim(0, 30)
plt.ylim(0, 15000)
plt.xticks(np.arange(0, 31, 5))
plt.yticks(np.arange(0, 15000, 1000))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/rowscols_vs_numel_omega2.png')
plt.show()
"""
#-------------------------------------------------------------------------------------------------------------------
"""
- nCliques vs numEl
Explain that omega doesn't affect the number of cliques
Splitting only, show how the number of cliques can increase with the element size
Explain since these are random problems, the number of cliques found won't be perfectly consistent/monotime
Increasing numEl, more cliques
"""

"""
# Single plot
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df = splittingMetaData.loc[splittingMetaData["omega"]==1]
plt.plot(df['num_el_real'], df['n_cliques'], '+--', linewidth=1, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Number of Cliques vs Number of Independent Variables", fontproperties=titleDict, y=0.95)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("# Cliques", fontsize=10)

# Limits and interval ticks
plt.xlim(0, 205)
plt.ylim(0, 102)
plt.xticks(np.arange(0, 205, 20))
plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/ncliques_vs_numel.png')
plt.show()
"""
#-------------------------------------------------------------------------------------------------------------------
"""
- Primal Residual vs Iteration
- Dual Residaul vs Iteration. 
Plot for splitting and no-splitting, plot several images for different problem sizes
"""

"""
# Plot for omega = 1, numel = 6
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['6_1']['i'].iloc[2:], noSplittingIterationData['6_1']['primal_residual'].iloc[2:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['6_1']['i'].iloc[2:], splittingIterationData['6_1']['primal_residual'].iloc[2:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['6_1']['i'].iloc[2:], noSplittingIterationData['6_1']['dual_residual'].iloc[2:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['6_1']['i'].iloc[2:], splittingIterationData['6_1']['dual_residual'].iloc[2:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 10 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_6_1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 1, numel = 21
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['21_1']['i'].iloc[1:], noSplittingIterationData['21_1']['primal_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['21_1']['i'].iloc[1:], splittingIterationData['21_1']['primal_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['21_1']['i'].iloc[1:], noSplittingIterationData['21_1']['dual_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['21_1']['i'].iloc[1:], splittingIterationData['21_1']['dual_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 40 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_21_1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 1, numel = 51
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['51_1']['i'].iloc[2:], noSplittingIterationData['51_1']['primal_residual'].iloc[2:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['51_1']['i'].iloc[2:], splittingIterationData['51_1']['primal_residual'].iloc[2:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['51_1']['i'].iloc[2:], noSplittingIterationData['51_1']['dual_residual'].iloc[2:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['51_1']['i'].iloc[2:], splittingIterationData['51_1']['dual_residual'].iloc[2:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 100 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_51_1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 1, numel = 101
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['101_1']['i'].iloc[1:], noSplittingIterationData['101_1']['primal_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['101_1']['i'].iloc[1:], splittingIterationData['101_1']['primal_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['101_1']['i'].iloc[1:], noSplittingIterationData['101_1']['dual_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['101_1']['i'].iloc[1:], splittingIterationData['101_1']['dual_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 200 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_101_1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 2, numel = 6
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['6_2']['i'].iloc[1:], noSplittingIterationData['6_2']['primal_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['6_2']['i'].iloc[1:], splittingIterationData['6_2']['primal_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['6_2']['i'].iloc[1:], noSplittingIterationData['6_2']['dual_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['6_2']['i'].iloc[1:], splittingIterationData['6_2']['dual_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$, 10 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_6_2.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 2, numel = 15
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['15_2']['i'].iloc[2:], noSplittingIterationData['15_2']['primal_residual'].iloc[2:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['15_2']['i'].iloc[2:], splittingIterationData['15_2']['primal_residual'].iloc[2:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['15_2']['i'].iloc[2:], noSplittingIterationData['15_2']['dual_residual'].iloc[2:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['15_2']['i'].iloc[2:], splittingIterationData['15_2']['dual_residual'].iloc[2:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$, 28 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_15_2.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 3, numel = 4
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['4_3']['i'].iloc[1:], noSplittingIterationData['4_3']['primal_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['4_3']['i'].iloc[1:], splittingIterationData['4_3']['primal_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

plt.plot(noSplittingIterationData['4_3']['i'].iloc[1:], noSplittingIterationData['4_3']['dual_residual'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['4_3']['i'].iloc[1:], splittingIterationData['4_3']['dual_residual'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Primal and Dual Residual vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=3$, 6 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Primal/Dual Residual", fontsize=10)
plt.legend(["Primal Residual - No Splitting", "Primal Residual - Splitting", "Dual Residual - No Splitting", "Dual Residual - Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/residuals_vs_i_4_3.png')
plt.show()
"""
#-------------------------------------------------------------------------------------------------------------------
"""
- Objective cost vs Iteration
Plot for splitting and no-splitting, for a few different problem sizes, to see/prove that they converge to the 
same answer and observe the speeds
"""

"""
# Plot for omega = 1, numel = 101
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['101_1']['i'].iloc[1:], noSplittingIterationData['101_1']['objective_cost'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['101_1']['i'].iloc[1:], splittingIterationData['101_1']['objective_cost'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Cost Function vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 200 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Cost Function", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/cost_vs_i_101_1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 1, numel = 51
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['51_1']['i'].iloc[1:], noSplittingIterationData['51_1']['objective_cost'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['51_1']['i'].iloc[1:], splittingIterationData['51_1']['objective_cost'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Cost Function vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 100 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Cost Function", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/cost_vs_i_51_1.png')
plt.show()
"""
###################################################################################################################
"""
# Plot for omega = 1, numel = 11
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
plt.plot(noSplittingIterationData['11_1']['i'].iloc[1:], noSplittingIterationData['11_1']['objective_cost'].iloc[1:], 'o--', linewidth=1, markersize=1)
plt.plot(splittingIterationData['11_1']['i'].iloc[1:], splittingIterationData['11_1']['objective_cost'].iloc[1:], 'x--', linewidth=1, markersize=1)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Cost Function vs Iteration", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 20 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Cost Function", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/cost_vs_i_11_1.png')
plt.show()
"""
#-------------------------------------------------------------------------------------------------------------------
"""
- Total time vs Problem Size
Plot comparison for splitting vs no splitting. Will need to process data to gather the total time for each run
Also might need to parse problem size, number of elements and maybe have another one for each omega
"""

# Plot for omega = 1, numel = 11
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df = noSplittingMetaData.loc[noSplittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==1]
plt.plot(df['num_el'], df['total_time'], 'o--', linewidth=1, markersize=1)
plt.plot(df2['num_el'], df2['total_time'], 'o--', linewidth=1, markersize=1)

# Title
# titleDict = { 'size': 12, 'weight': 'semibold'}
# plt.suptitle("Cost Function vs Iteration", fontproperties=titleDict, y=0.96)
# subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
# plt.title(r"$\omega=1$, 20 Independent Variables", fontdict=subtitleDict, pad=7)

# Lables and legend
# plt.xlabel("Iteration", fontsize=10)
# plt.ylabel("Cost Function", fontsize=10)
# plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
# plt.savefig('./images/cost_vs_i_11_1.png')
plt.show()