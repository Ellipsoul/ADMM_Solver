import pandas as pd
import pickle
import timeit

# Some useful prefixes
prefix = "../results"

noSplittingIterationsPrefix = f"{prefix}/no_splitting/nosplitting_iterations_"
noSplittingMetaPrefix = f"{prefix}/no_splitting/nosplitting_meta_"

splittingIterationsPrefix = f"{prefix}/splitting/splitting_iterations_"
splittingMetaPrefix = f"{prefix}/splitting/splitting_meta_"

# Initialise dictionaries of data
# Keys for each dataset will be stored with this form '5_2'
noSplittingIterationData = {}
noSplittingMetaData = {}
splittngIterationData = {}
splittingMetaData = {}

# Define array of keys for all trials that were run
keyArray = []
keyArray.extend([f"{i}_3" for i in range(3, 5)])        # All omega=3 trials
keyArray.extend([f"{i}_2" for i in range(3, 16)])       # All omega=2 trials
keyArray.extend([f"{i}_1" for i in range(3, 21)])       # All omega=1 trials
keyArray.extend([f"{i}_1" for i in range(21, 50, 2)])
keyArray.extend([f"{i}_1" for i in range(51, 80, 5)])
keyArray.extend([f"{i}_1" for i in range(81, 102, 10)])

# Import all the data into the dictionaries
for key in keyArray:
    noSplittingIterationData[key] = pd.read_csv(f"{noSplittingIterationsPrefix}{key}.csv")
    noSplittingMetaData[key] = pd.read_csv(f"{noSplittingMetaPrefix}{key}.csv")
    
    splittngIterationData[key] = pd.read_csv(f"{splittingIterationsPrefix}{key}.csv")
    splittingMetaData[key] = pd.read_csv(f"{splittingMetaPrefix}{key}.csv")

# Concatenate data (really useful!)
noSplittingMetaData = pd.concat(noSplittingMetaData.values())
splittingMetaData = pd.concat(splittingMetaData.values())

# Create new column with real number of elements
noSplittingMetaData['num_el_real'] = noSplittingMetaData['num_el'].apply(lambda i: 2*(i-1))
splittingMetaData['num_el_real'] = splittingMetaData['num_el'].apply(lambda i: 2*(i-1))

# Store the data to be imported (it's slow because of pickling though)
with open("allData.pk", 'wb') as fi:
    pickle.dump([noSplittingIterationData, noSplittingMetaData, splittngIterationData, splittingMetaData], fi)
