import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import timeit

# Loading the pickled data back
with open('allData.pk', 'rb') as fi:
    noSplittingIterationData, noSplittingMetaData, splittngIterationData, splittingMetaData = pickle.load(fi)

"""
Plots to make

- nCliques vs numEl
Explain that omega doesn't affect the number of cliques
Splitting only, show how the number of cliques can increase with the element size
Explain since these are random problems, the number of cliques found won't be perfectly consistent/monotime
Increasing numEl, more cliques

- Average clique size vs numEl (maybe)
Splitting only, show the average number of rows and columns in a clique as problem size increases

- Primal Residual vs Iteration
- Dual Residaul vs Iteration. 
Plot for splitting and no-splitting, plot several images for different problem sizes

- Objective cost vs Iteration
Plot for splitting and no-splitting, for a few different problem sizes, to see/prove that they converge to the 
same answer and observe the speeds

- Total time vs Problem Size
Plot comparison for splitting vs no splitting. Will need to process data to gather the total time for each run
Also might need to parse problem size, number of elements and maybe have another one for each omega

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

"""
- Problem size vs Number of Elements
Might be useful as a start to show how big the A matrices get as number of elements increase
Plot for rows and columns on the same graph
Also shows how quickly increasing omega increases the problem size 
"""

# plt.figure(figsize=(7, 5), dpi=120)  # Dimensions in inches

# New figure
# plt.figure(figsize=(7, 5), dpi=120)  # Dimensions in inches

# Plot the desired points
# for frame in [noSplittingIterationData["5_1"], noSplittingIterationData["10_1"], noSplittingIterationData["15_1"]]:
#     plt.plot(frame['i'], frame['objective_cost'])

# Title
# fontdict = { 'fontsize': 20, 'fontweight': 'bold', 'verticalalignment': 'baseline'}
# plt.title("Test Title", fontdict=fontdict, pad=15)

# Lables and legend
# plt.xlabel("Iteration", fontsize=10)
# plt.ylabel("Objective Cost", fontsize=10)
# plt.legend(["5", "10", "15"], loc=0)

# Limits and interval ticks
# plt.xlim(0, 10000)
# plt.ylim(-10, -4)
# plt.xticks(np.arange(0, 10001, 1000))
# plt.yticks(np.arange(-10, -3, 1))

# plt.grid(linestyle="--", linewidth=0.5)
# plt.savefig('test.png', bbox_inches='tight')
# plt.show()
