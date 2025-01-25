import shapefile
import numpy as np
import pandas as pd
from scipy import stats
from getNeighbors import getNeighborsAreaContiguity,extractCentroidsFromShapefile, kNearestNeighbors

# Load shapefile data
sf = shapefile.Reader("input/Hex437.shp") # Synthetic data
#sf = shapefile.Reader("input/NPA_2014_Mecklenburg.shp") # Case study with Mecklenburg County 
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

# Define Tao & Thill's local BiT statistic
def calculateBivariateT(ikey, keyList, dataDictionary):
    """
    Calculate the local bivariate BiT statistic for a given unit and its neighbors.
    ikey: ID of the focal unit
    keyList: List of neighboring unit IDs
    dataDictionary: Standardized variable dictionary
    """
    sum_value = 0
    for j in keyList:
        if j in dataDictionary:
            diff1 = dataDictionary[ikey][0] - dataDictionary[j][1]
            diff2 = dataDictionary[ikey][1] - dataDictionary[j][0]
            sum_value += np.sqrt(diff1**2 * diff2**2)
    return sum_value

# Compute BiT for each spatial unit
BTValues = {}
result = []
for x in areaKeys:
    keyList = neighbors[x]
    currentT = calculateBivariateT(x, keyList, dataDictionary_std)
    result.append(currentT)
    BTValues[x] = currentT

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
        randomT = calculateBivariateT(x, permKey, dataDictionary_std)
        if BTValues[x] > randomT:
            betterClusters += 1
    p = (betterClusters + 1) / 1000.0 # The most important result. The highest/lowest ones correspond to negative/positive clusters
    plist.append(p)
    pvalue.append(p if p < 0.5 else 1 - p) # Adjust negative cluster's p-value to small values

# Identify patterns at different significance levels
idx0001 = []
idx001 = []
idx005 = []
for x in areaKeys:
    if plist[x] <= 0.001:
        idx0001.append('pos_cluster')
    elif plist[x] >= 0.999:
        idx0001.append('neg_cluster')
    else:
        idx0001.append('NS')

    if plist[x] <= 0.01:
        idx001.append('pos_cluster')
    elif plist[x] >= 0.99:
        idx001.append('neg_cluster')
    else:
        idx001.append('NS')

    if plist[x] <= 0.05:
        idx005.append('pos_cluster')
    elif plist[x] >= 0.95:
        idx005.append('neg_cluster')
    else:
        idx005.append('NS')

# Save results to a DataFrame and export to CSV
df = pd.DataFrame({
    'OBJECTID': id + 1,  # Add 1 to IDs for compatibility
    'Cluster': Cluster,
    'Edge': Edge,
    'var1': var1,  # Original Variable 1 values
    'var2': var2,  # Original Variable 2 values
    'BiT': result,  # Computed BiT statistic
    'p_sim': plist,  # Pseudo p-values
    'p_value': pvalue,  # Adjusted p-values
    'pattern005': idx005,  # Patterns at 0.05 significance level
    'pattern001': idx001,  # Patterns at 0.01 significance level
    'pattern0001': idx0001  # Patterns at 0.001 significance level
})

df.to_csv("result/BiT_Hex437_XY_Quali.csv", index=False)
#df.to_csv("result/BiT_Hex437_Z2Z_Quali_10_50_90.csv", index=False)
#df.to_csv("result/BiT_Meck_Bac_ELem.csv", index=False)
print("Processing Complete.")
