""" local Lee's L (Lee 2001)
Lee, S. I. (2001). Developing a bivariate spatial association measure: an integration of Pearson's r and Moran's I. Journal of Geographical Systems, 3(4), 369-385."""

import core.shapefile
import numpy as np
import pandas as pd
from scipy import stats
from core.getNeighbors import getNeighborsAreaContiguity,extractCentroidsFromShapefile, kNearestNeighbors
from core.spatstats import calculateLocalL

# Load shapefile data
sf = core.shapefile.Reader("input/Hex437.shp") # Synthetic data
#sf = core.shapefile.Reader("input/NPA_2014_Mecklenburg.shp") # Case study with Mecklenburg County 
shapes = sf.shapes()

# Prepare AREAS input for Queen's and Rook's contiguity
AREAS = [[shape.points] for shape in shapes]  # Ensure proper structure for AREAS

# Calculate neighbors using Queen's and Rook's contiguity
Wqueen, Wrook = getNeighborsAreaContiguity(AREAS)
neighbors = Wqueen  # Use Queen's contiguity for further analysis

"""
# Use k nearest neighbors instead of contiguity
shapefile_path = "input/Hex437.shp"
centroids = extractCentroidsFromShapefile(shapefile_path)

# Compute k-nearest neighbors (adjust k as needed)
k = 5  # Number of nearest neighbors
neighbors = kNearestNeighbors(centroids, k=k)
"""

# Load and extract raw data from CSV
data = pd.read_csv("input/Hex437noNorm.csv") # Synthetic data
#data = pd.read_csv("input/CLT_Data_Bac_Elem.csv") # Case study with Mecklenburg County 

id = data.FID  # Unique IDs for spatial units
var1 = data.X  # Variable 1 values in synthetic data
var2 = data.Y  # Variable 2 values in synthetic data
#var1 = data.Z2_Value90_50  # Variable 1 values in synthetic data
#var2 = data.Z_Value90_50  # Variable 2 values in synthetic data
Cluster = data.Cluster  # Cluster membership (targeted special zones in synthetic data)
Edge = data.Edge  # Edge of special zones, useful for edge effect considerations


#var1 = data.Bac  # Variable 1 values in Mecklenburg data: % of adults with bachelor's degree or higher
#var2 = data.Elem  # Variable 2 values in synthetic data: average test scores of elementary school students


# Create a dictionary mapping IDs to variables
dataDictionary = {int(b): [var1[a], var2[a]] for a, b in enumerate(id)}
areaKeys = list(dataDictionary.keys())

# Standardize variables
var1_std = [(val - np.mean(var1)) / np.std(var1) for val in var1]
var2_std = [(val - np.mean(var2)) / np.std(var2) for val in var2]
dataDictionary_std = {int(b): [var1_std[a], var2_std[a]] for a, b in enumerate(id)}


# Compute (bivariate) L for each spatial unit
BLValues = {}
result = []
for x in areaKeys:
    keyList = neighbors[x]
    currentBL = calculateLocalL(x, keyList, dataDictionary_std)
    result.append(currentBL)
    BLValues[x] = currentBL

# Monte Carlo permutation for significance test
plist = []
pvalue = []
dataLength = len(shapes)

for x in areaKeys:
    Nlist = list(range(dataLength))
    betterClusters = 0
    number = len(neighbors[x])
    Nlist.remove(x)
    for _ in range(999):  # Perform 999 random permutations
        permKey = np.random.choice(Nlist, number, replace=False)
        randomBL = calculateLocalL(x, permKey, dataDictionary_std)
        if BLValues[x] > randomBL:
            betterClusters += 1
    p = (betterClusters + 1) / 1000.0 # The most important result. The highest/lowest ones correspond to positive/negative clusters of bivariate spatial association
    plist.append(p)
    pvalue.append(p if p < 0.5 else 1 - p) # Adjust positive cluster's p-value to small values

# Identify patterns at different significance levels
idx = []
dataMean = np.mean(np.double(var1))
dataMean2 = np.mean(np.double(var2))

for x in areaKeys:
    if plist[x] >= 0.95: # Modify to 0.99 (0.999) for p-value < 0.01 (0.001) level
        if dataDictionary[x][0] < dataMean and dataDictionary[x][1] < dataMean2:
            idx.append('LL')
        elif dataDictionary[x][0] > dataMean and dataDictionary[x][1] > dataMean2:
            idx.append('HH')
        else:
            idx.append('OtherBgL')
    elif plist[x] <= 0.05:
        if dataDictionary[x][0] < dataMean and dataDictionary[x][1] > dataMean2:
            idx.append('LH')
        elif dataDictionary[x][0] > dataMean and dataDictionary[x][1] < dataMean2:
            idx.append('HL')
        else:
            idx.append('OtherSmL')
    else:
        idx.append('NS')


# Save results to a DataFrame and export to CSV
df = pd.DataFrame({
    'OBJECTID': id + 1,  # Add 1 to IDs for compatibility
    'Cluster': Cluster,
    'Edge': Edge,
    'var1': var1,  # Original Variable 1 values
    'var2': var2,  # Original Variable 2 values
    'L': result,  # Computed Lee's L statistic
    'p_sim': plist,  # Pseudo p-values
    'p_value': pvalue,  # Adjusted p-values
    'pattern': idx,  # Patterns that passed the significant test
})

df.to_csv("result/LeeL_Hex437_XY_Quali.csv", index=False)
#df.to_csv("result/BiT_Hex437_Z2Z_Quali_10_50_90.csv", index=False)
#df.to_csv("result/BiT_Meck_Bac_ELem.csv", index=False)
print("Processing Complete.")
