### local BiT: A Reciprocal Statistic for Detecting the Full Range of Local Patterns of Bivariate Spatial Association


| `localBiMoranI.py`   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobyellow/localBiT/blob/main/localBiMoranI.ipynb)   |
| `localBiT.py`        | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobyellow/localBiT/blob/main/localBiT.ipynb)        |
| `localLeeL.py`       | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobyellow/localBiT/blob/main/localLeeL.ipynb)       |
| `localMultiGearyC.py`| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bobyellow/localBiT/blob/main/localMultiGearyC.ipynb)|



Bivariate spatial association (BSA) refers to the degree of similarity between two independently measured values observed in spatial proximity. 
The concept of BSA is an extension of univariate spatial association (also referred to as spatial autocorrelation), which is encapsulated by Tobler’s first law of geography: “everything is related to everything else, but near things are more related than distant things” (Tobler 1970).
Likewise, BSA also centers on these two types of similarities. It refers to the similarity between the association trends of two variables in the same geographic neighborhood globally, that is, over the full spatial extent of the study. 

The figure below depicts two variables in the same space (they are represented in different visual frames for clarity). Specifically, variable x has high, high, medium-high values in regions A, B, and C, respectively; variable y has high, low, and medium-high values in the same three localized regions. In line with the principle of BSA, strong positive association is expected to exist in region A, where the two variables have similar trends, as they both have high values. Also, strong negative BSA is expected in region B, where the two variables show opposite local trends. Notwithstanding that neither x nor y holds extreme values in region C, they are still expected to have strong positive BSA given their highly similar local trends. However, **while existing local statistics such as the bivariate local Moran’s I (Anselin et al. 2002) and the local Lee’s L statistic (Lee 2001) can correctly detect patterns of spatial association in regions A and B, they would miss association patterns in region C.** This can be traced to their formulation as the product of x or y (or their spatial lags) in spatial proximity, which inevitably boosts extreme-value combinations. **We argue that patterns formed by non-extreme variable values deserve to be discovered and discussed** like values in the tail ends of distributions, provided that they happen in spatial proximity. The goal of this study, therefore, is to propose and demonstrate a new statistic for detecting local patterns of BSA that is not predicated by local univariate measurement values with respect to their global first moment. 
![Fig1](https://github.com/user-attachments/assets/54b5ee84-86ca-49b4-9ee6-3dbb36a71f97)

In the following simple example, Zones 1, 5, and 9 are expected to have strong positive BSA due to high similarity between the values taken by two variables in spatial proximity; Zones 3 and 7 are expected to have strong negative BSA due to the opposite local trends of the two variables in the vicinity of these zones; Zones 2, 4, 6, and 8 are not expected to exhibit significant patterns because one variable takes extreme values while the other is medium; they are neither highly similar nor highly dissimilar in spatial proximity.
![Fig2](https://github.com/user-attachments/assets/860eb031-201c-43e0-86f9-6aeeddbb6f31)

The results show that all three bivariate statistics (local BiT, local bivariate Moran's I, and local Lee's L) are effective at detecting strong BSA patterns formed by extreme values. **But only local BiT can detect strong positive BSA patterns formed by non-extreme values (MM) in Zone 5**, whereas the other two statistics cannot. Also, the multivariate local Geary’s C (Anselin 2019) is proven to have distinct purpose and functionality from the other statistics, as it measures the joint spatial association of all variables while overlooking their mutual relationships that the BSA hinges upon!
![Fig3](https://github.com/user-attachments/assets/0a6c3007-f7d5-4d96-891a-b725ae85e8a9)


### Abstract of the paper
Bivariate spatial association is the relationship between two variables in spatial proximity.  Observation of strong bivariate spatial association rests on the similarity of two variables in the same geographic neighborhood, and it should not be conditioned by the concentration of extreme values. However, **existing spatial statistical methods put disproportionate emphasis on patterns formed by extreme values**, such as the so-called “high-high”, “low-low”, “high-low”, and “low-high” patterns. The consequence is that patterns of strong bivariate spatial association formed by non-extreme values are often ignored, as if they were “less interesting” or did not exist. In this study, we solve this issue by proposing a new exploratory local spatial statistic for detecting the full range of bivariate spatial association, dubbed local BiT. In comparison with the widely adopted bivariate local Moran’s I and local Lee’s L, **local BiT can detect patterns of bivariate spatial association regardless of whether the variable values are high, low, or anywhere in-between**. In addition, its reciprocal design guarantees that the order of two variables in calculations does not lead to different results. Moreover, it avoids false positive errors arising when one variable has extreme value while the other is non-extreme. Properties of the new statistic are studied on synthetic datasets. A case study is conducted in Mecklenburg County, North Carolina, to examine the spatial association between adults’ educational attainment and elementary school students’ academic performance. This study of spatial demographics and human capital demonstrates the differences and value of the BiT over other methods.

### Reference:
(**This paper**) Ran Tao, Jean-Claude Thill. A Reciprocal Statistic for Detecting the Full Range of Local Patterns of Bivariate Spatial Association. _Annals of the American Association of Geographers_. (forthcoming)

Anselin, L., Syabri, I., & Smirnov, O. (2002). Visualizing multivariate spatial correlation with dynamically linked windows. In Proceedings, CSISS Workshop on New Tools for Spatial Data Analysis, Santa Barbara, CA.

Anselin, L. (2019). A local indicator of multivariate spatial association: extending Geary's C. Geographical Analysis, 51(2), 133-150. 

Lee, S. I. (2001). Developing a bivariate spatial association measure: an integration of Pearson's r and Moran's I. Journal of Geographical Systems, 3(4), 369-385.


