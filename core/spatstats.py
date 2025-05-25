import numpy as np

# Define Tao & Thill's local BiT statistic
def calculateBivariateT(ikey, keyList, dataDictionary):
    """
    Calculate the local bivariate BiT statistic for a given unit and its neighbors.
    This implementation assumes that both variable x and y have been standardized, so their mean equals 0, std equals 1.
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
    BT = sum_value
    return BT



# Define the bivariate local Moran's I statistic
def calculateBivaraiteMoranI(ikey, keyList, dataDictionary):
    """
    This function returns the local bivaraite Moran's I statistic a given region.
    This implementation assumes that both variable x and y have been standardized, so their mean equals 0, std equals 1.
    keyList is the list of the keys of i's neighbors
    dataLength is the total number of input data units
    dataDictionary has multivariates {key:[v1,v2,...,vk]}
    numVar is the total number of variables
    """
    sum = 0
    dataDictionary1 = { key:value for key, value in dataDictionary.items()}
    neighborNumber = len(keyList)
    for j in keyList:
       if j in dataDictionary1.keys():
            #standardize variable j
            sum += dataDictionary1[j][1] 
            #for non-binary wij, work as:
            #sum = sum + std_j_value * wij

    #To row standardize: sum of wij for each i equals 1
    if neighborNumber!=0:
        sum = sum/neighborNumber
    bi = sum * dataDictionary1[ikey][0]
    BI = bi
  
    return BI



# Define the local Lee's L statistic
def calculateLocalL(ikey, keyList, dataDictionary):
    """
    This function returns the local Lee's L statistic by Lee 2001
    This implementation assumes that both variable x and y have been standardized, so their mean equals 0, std equals 1.
    ikey is i in the equation
    keyList is the list of the keys of i's neighbors
    dataLength is the total number of input data units
    dataDictionary has multivariates {key:[v1,v2,...,vk]}
    numVar is the total number of variables
    """
    sum1 = 0
    sum2 = 0
    dataDictionary1 = { key:value for key, value in dataDictionary.items()}
    neighborNumber = len(keyList)

    for j in keyList:
       if j in dataDictionary1.keys():
             sum1 = sum1 + dataDictionary1[j][0]
             sum2 = sum2 + dataDictionary1[j][1]
            
    L = sum1 * sum2
    return L



# Define the multivariate local Geary's C statistic
def calculateMultiGearyC(ikey, keyList, dataDictionary, dataDictionaryPer, numVar):
    """
    This function returns the local Geary's c statistic for a given region.
    keyList is the list of the keys of i's neighbors.
    dataDictionary has multivariates {key:[v1,v2,...,vk]}.
    numVar is the total number of variables.
    """
    total_sum = 0
    for i in range(numVar):
        sum_var = 0
        data_values = list(dataDictionaryPer.values())
        dataSum = np.sum(np.array([item[i] for item in data_values], dtype=float))
        dataMean = np.mean(np.array([item[i] for item in data_values], dtype=float))
        dataStd = np.std(np.array([item[i] for item in data_values], dtype=float))
        
        # Check if keyList is empty
        if isinstance(keyList, np.ndarray):
            if keyList.size == 0:
                continue
        else:
            if len(keyList) == 0:
                continue
        
        for j in keyList:
            std_i_value = (dataDictionary[ikey][i] - dataMean) / dataStd
            std_j_value = (dataDictionaryPer[j][i] - dataMean) / dataStd
            sum_var += (std_i_value - std_j_value) ** 2
        total_sum += sum_var

    MC = total_sum / numVar
    return MC


# Define the local G statistic
def calculateGetisG(keyList, dataMean, dataStd, dataDictionary, dataLength):
    """
    This function returns the local G statistic a given region.
    keyList is the list of keys of neighbors
    dataLength is the total number of input data units
    """
    sum = 0
    dataDictionary1 = { key:value for key, value in dataDictionary.items()}
    neighborNumber = len(keyList)
    for i in keyList:
        sum += np.double(dataDictionary1[i])
    numerator = sum - (dataMean * neighborNumber)
    denominator = dataStd * ((float(dataLength * neighborNumber - (neighborNumber ** 2)) / (dataLength - 1)) ** 0.5)

    G = (np.double(numerator))/(np.double(denominator))
    return G



# Define the (univariate) local Moran's I statistic
def calculateMoranI(ikey, keyList, dataMean, dataStd, dataDictionary, dataLength):
    """
    This function returns the local Moran's I statistic a given region.
    keyList is the list of the keys of i's neighbors
    dataLength is the total number of input data units
    """
    sum = 0
    dataDictionary1 = { key:value for key, value in dataDictionary.items()}
    neighborNumber = len(keyList)
    for j in keyList:
        sum += np.double((dataDictionary1[j])- dataMean)
    numerator = dataLength*(dataDictionary1[ikey] - dataMean)*sum
    denominator = dataStd ** 2
    #To row standardize: sum of wij for each i equals 1
    denominator = denominator * neighborNumber
    
    I = (np.double(numerator))/(np.double(denominator))
    return I


# Define the (univariate) local Geary's C statistic
def calculateGearyC(ikey, keyList, dataDictionary):
    """
    This function returns the local Geary's c statistic a given region.
    keyList is the list of the keys of i's neighbors
    dataLength is the total number of input data units
    """
    sum = 0
    dataDictionary1 = { key:value for key, value in dataDictionary.items()}
    neighborNumber = len(keyList)
    for j in keyList:
         sum += np.double((dataDictionary1[ikey]- dataDictionary1[j])**2)
    C = sum
    return C
