import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import timeit

# Loading the pickled data back
with open('allData.pk', 'rb') as fi:
    noSplittingIterationData, noSplittingMetaData, splittingIterationData, splittingMetaData = pickle.load(fi)


# Start of Plots!
#-------------------------------------------------------------------------------------------------------------------
"""
- Problem size vs Number of Elements
Might be useful as a start to show how big the A matrices get as number of elements increase
Plot for rows and columns on the same graph
Also shows how quickly increasing omega increases the problem size 
"""


# Figure 1
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 2
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
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- nCliques vs numEl
Explain that omega doesn't affect the number of cliques
Splitting only, show how the number of cliques can increase with the element size
Explain since these are random problems, the number of cliques found won't be perfectly consistent/monotime
Increasing numEl, more cliques
"""


# Figure 3
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
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Primal Residual vs Iteration
- Dual Residaul vs Iteration. 
Plot for splitting and no-splitting, plot several images for different problem sizes
"""


# Figure 4
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 5
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 6
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 7
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 8
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 9
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 10
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
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Objective cost vs Iteration
Plot for splitting and no-splitting, for a few different problem sizes, to see/prove that they converge to the 
same answer and observe the speeds
"""


# Figure 11
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 12
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
# plt.show()
plt.close()

###################################################################################################################

# Figure 13
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
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Total time vs Problem Size
Plot comparison for splitting vs no splitting. Will need to process data to gather the total time for each run
Also might need to parse problem size, number of elements and maybe have another one for each omega
"""


# Figure 14
# Plot for omega = 1
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[noSplittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==1]
plt.plot(df1['num_el_real'], df1['total_time'], '+--', linewidth=2, markersize=4)
plt.plot(df2['num_el_real'], df2['total_time'], '+--', linewidth=2, markersize=4)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Total CPU Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("CPU Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/cputime_vs_numel_omega1.png')
# plt.show()
plt.close()

###################################################################################################################

# Figure 15
# Plot for omega = 2
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[noSplittingMetaData["omega"]==2]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==2]
plt.plot(df1['num_el_real'], df1['total_time'], '+--', linewidth=2, markersize=4)
plt.plot(df2['num_el_real'], df2['total_time'], '+--', linewidth=2, markersize=4)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Total CPU Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("CPU Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/cputime_vs_numel_omega2.png')
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Setup time vs numEl
Splitting and nosplitting, maybe plot on separate graphs
Explain that setup is negligible for nosplitting since cliques don't need to be detected
Also a lot more overhead with moving the cliques to the 
Observe the additional time required for clique detection as it increases
"""


# Figure 16
# Plot for omega = 1
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[noSplittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==1]
plt.plot(df1['num_el_real'], df1['setup_time'], '+--', linewidth=1, markersize=4)
plt.plot(df2['num_el_real'], df2['setup_time'], '+--', linewidth=1, markersize=4)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Setup Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Setup Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/setuptime_vs_numel_omega1.png')
# plt.show()
plt.close()

###################################################################################################################

# Figure 17
# Plot for omega = 2
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[noSplittingMetaData["omega"]==2]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==2]
plt.plot(df1['num_el_real'], df1['setup_time'], '+--', linewidth=1, markersize=4)
plt.plot(df2['num_el_real'], df2['setup_time'], '+--', linewidth=1, markersize=4)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Setup Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Setup Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/setuptime_vs_numel_omega2.png')
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- nxTime vs numEl
Splitting only. Might be useful to see exactly how much time this part takes
Maybe plot with the setup time to see how much of the setup time is nx Time
"""


# Figure 18
# Single plot (omega=1,2,3 on the same plot, splitting only)
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = splittingMetaData.loc[splittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==2]
df3 = splittingMetaData.loc[splittingMetaData["omega"]==3]

plt.plot(df1['num_el_real'], df1['nx_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['nx_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df3['num_el_real'], df3['nx_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Clique Detection Time vs Number of Independent Variables", fontproperties=titleDict, y=0.94)
# subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
# plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Clique Detection Time (s)", fontsize=10)
# plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/nxtime_vs_numel.png')
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Update y Time vs numEl
Splitting and no splitting (on same plot). Do omega=1 and 2 on different plots
Splitting still takes more time because it needs to update the local vector within each clique
"""


# Figure 19
# Plot for omega=1
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[splittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==1]

plt.plot(df1['num_el_real'], df1['update_y_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['update_y_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Update Y Vector Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Update Y Vector Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/updateytime_vs_numel_omega1.png')
# plt.show()
plt.close()

###################################################################################################################

# Figure 20
# Plot for omega=2
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[splittingMetaData["omega"]==2]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==2]

plt.plot(df1['num_el_real'], df1['update_y_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['update_y_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Update Y Vector Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Update Y Vector Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/updateytime_vs_numel_omega2.png')
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Update z Time vs numEl
Splitting and no splitting. Same, this projection time will be different since splitting has to do it for all cliques
Need to comment on the difference, and mention that with a for loop/non-parallelised it's definitely slower
"""


# Figure 21
# Plot for omega=1
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[splittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==1]

plt.plot(df1['num_el_real'], df1['update_z_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['update_z_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Update Z Vector Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Update Z Vector Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/updateztime_vs_numel_omega1.png')
# plt.show()
plt.close()

###################################################################################################################

# Figure 22
# Plot for omega=2
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[splittingMetaData["omega"]==2]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==2]

plt.plot(df1['num_el_real'], df1['update_z_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['update_z_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Update Z Vector Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Update Z Vector Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/updateztime_vs_numel_omega2.png')
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Update Lagrange time
Splitting and no splitting, same as above
"""


# Figure 23
# Plot for omega=1
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[splittingMetaData["omega"]==1]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==1]

plt.plot(df1['num_el_real'], df1['update_lagrange_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['update_lagrange_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Update Lagrange Multipliers Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Update Lagrange Multiplieres Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/updatelagrangetime_vs_numel_omega1.png')
# plt.show()
plt.close()

###################################################################################################################

# Figure 24
# Plot for omega=2
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
df1 = noSplittingMetaData.loc[splittingMetaData["omega"]==2]
df2 = splittingMetaData.loc[splittingMetaData["omega"]==2]

plt.plot(df1['num_el_real'], df1['update_lagrange_time'], '+--', linewidth=1.5, markersize=5)
plt.plot(df2['num_el_real'], df2['update_lagrange_time'], '+--', linewidth=1.5, markersize=5)

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("Update Lagrange Multipliers Time vs Number of Independent Variables", fontproperties=titleDict, y=0.96)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$", fontdict=subtitleDict, pad=7)

# Lables and legend
plt.xlabel("# Independent Variables", fontsize=10)
plt.ylabel("Update Lagrange Multipliers Time (s)", fontsize=10)
plt.legend(["No Splitting", "Splitting"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 205)
# plt.ylim(0, 102)
# plt.xticks(np.arange(0, 205, 20))
# plt.yticks(np.arange(0, 102, 10))

# Grid and housekeeping
plt.grid(linestyle="--", linewidth=0.5)
plt.savefig('./images/updatelagrangetime_vs_numel_omega2.png')
# plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------------------
"""
- Breakdown of time taken for each step (pie chart)
Splitting and no splitting. Really important to show which steps are taking the most time
Have an 'other' segment for anything that wasn't timed
Show a few pie charts for different problem sizes to observe the differences
"""

# Filtering columns for all pie charts
df1 = noSplittingMetaData[["num_el", "omega", "setup_time", "update_y_time", "update_z_time", "update_lagrange_time"]]
df2 = splittingMetaData[["num_el", "omega", "setup_time", "update_y_time", "update_s_time","update_z_time", "update_lagrange_time", "update_residual_time", "update_cost_time"]]


# Figure 25
# Plot for omega=1, numEl=6, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==6) & (df1['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 10 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_6_1_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 26
# Plot for omega=1, numEl=6, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==6) & (df2['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 10 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_6_1_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 27
# Plot for omega=1, numEl=21, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==21) & (df1['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.2f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 40 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_21_1_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 28
# Plot for omega=1, numEl=21, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==21) & (df2['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 40 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_21_1_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 29
# Plot for omega=1, numEl=51, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==51) & (df1['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.2f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 100 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_51_1_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 30
# Plot for omega=1, numEl=51, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==51) & (df2['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 100 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_51_1_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 31
# Plot for omega=1, numEl=101, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==101) & (df1['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.2f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 200 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_101_1_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 32
# Plot for omega=1, numEl=101, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==101) & (df2['omega']==1)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=1$, 200 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_101_1_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 33
# Plot for omega=2, numEl=6, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==6) & (df1['omega']==2)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.2f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$, 10 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_6_2_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 34
# Plot for omega=2, numEl=6, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==6) & (df2['omega']==2)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$, 10 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_6_2_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 35
# Plot for omega=2, numEl=15, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==15) & (df1['omega']==2)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.2f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$, 28 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_15_2_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 36
# Plot for omega=2, numEl=15, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==15) & (df2['omega']==2)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=2$, 28 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_15_2_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 37
# Plot for omega=3, numEl=4, no splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df1.loc[(df1['num_el']==4) & (df1['omega']==3)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update z', 'Update Lagrange']
explode = [0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.2f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=3$, 6 Independent Variables, No Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_4_3_nosplitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################

# Figure 38
# Plot for omega=3, numEl=4, splitting
plt.figure(figsize=(7, 5), dpi=120)

# Processing and plotting
array = df2.loc[(df2['num_el']==4) & (df2['omega']==3)].drop(['omega', 'num_el'], axis=1).to_numpy()[0]
labels = ['Setup', 'Update Y', 'Update s','Update z', 'Update Lagrange', 'Update Residual', 'Update Cost']
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(array, normalize=True, labels=labels, explode=explode, autopct='%.1f')

# Title
titleDict = { 'size': 12, 'weight': 'semibold'}
plt.suptitle("CPU Time Distribution", fontproperties=titleDict, y=0.95)
subtitleDict = { 'fontsize': 9, 'fontweight': 'normal', 'verticalalignment': 'baseline'}
plt.title(r"$\omega=3$, 6 Independent Variables, Splitting", fontdict=subtitleDict, pad=0)

# Grid and housekeeping
plt.savefig('./images/pie_4_3_splitting.png', bbox_inches='tight')
# plt.show()
plt.close()

###################################################################################################################