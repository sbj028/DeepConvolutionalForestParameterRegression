# DeepConvolutionalForestParameterRegression

# Datasets
Processed Sentinel-1 scenes for the Liwale AOI in Tanzania and the three Norwegian AOIs are openly available, and can be downloaded from ed [here](https://drive.google.com/drive/folders/1ZyxcArRPV6FfXzZsCIwSgECJCrx-AYxf).

The ALS-derived AGB prediction maps for the Tanzanian dataset are available on request from the corresponding author of _"Mapping and estimating forest area and aboveground biomass in miombo woodlands in Tanzania using data from airborne laser scanning, TanDEM-X, RapidEye, and global forest maps: A comparison of estimated precision"_, see [here](https://www.sciencedirect.com/science/article/abs/pii/S0034425716300062) for more infromation.  

The ALS-derived SV prediction maps for the Norwegian datasets are available on request from the corresponding author of _"Comparing the accuracies of forest attributes predicted from airborne laser scanning and digital aerial photogrammetry in operational forest inventories"_, see [here](https://www.sciencedirect.com/science/article/pii/S0034425719301178) for more information. 

The ground reference AGB data are part of the Tanzanian National Forest Inventory program. They are confidential and, therefore, not publicly available. The ground reference SV data from Norway are private property and, therefore, not publicly available. Thus are the masks representing the ground reference target data, and the coverage of the ALS-derived SV prediction maps for the Norwegian datasets not publicly available.

## Tanzanian models: 
The Tanzanian models were trained using a false-colour (RGB) Sentinel-1 image patches, i.e. each having the shape 64x64x3. In total, the number of target channels were two, stacked as follows `[ALS-derived prediction maps, ground reference target mask]`. See [here](https://arxiv.org/abs/2306.11103) for the paper and reference to the input and target data. 

## Norwegian models: 
The Norwegian models were trained using 9-channel Sentinel-1 image patches, i.e. each having the shape 64x64x9. In total, the number of target channels were three, stacked as follows `[ALS-derived prediction maps, timber volume mask, ground reference target mask]`. See [here](https://arxiv.org/abs/2306.11103) for the paper and reference to the input and target data.


# Source code
The source code for the paper _"Forest Parameter Prediction by Multiobjective Deep Learning of Regression Models Trained with Pseudo-Target Imputation"_, see [here](https://arxiv.org/abs/2306.11103) for the paper, can be found in the `python` directory.

