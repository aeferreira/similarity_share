# MetAnalyst

This folder contain the example from MetaboAnalyst 4.0 used in comparing the pre-treatments developed in scaling.py (and in the Metabolinks Python package) and the pre-treatments made by MetaboAnalyst 4.0. The data is available in the statistical analysis section of MetaboAnalyst, being the test data labelled as MS Peak Lists (Three-column LC-MS peak list files for 12 mice spinal cord samples). Here, are a plethora of files with different pre-treatments made to the original files:

#### Example dataset used (from MetaboAnalyst 4.0):

- MetAna_Original.csv - Original example dataset used without pre-treatments.

#### Missing Value Imputation

- MetAna_Imputed.csv - Only Missing Value Imputation by half of the minimum value in the dataset was applied (without peak filtering).
- MetAna_Imputed2.csv - Filtering of peaks with more than 50% of missing values followed by Missing Value Imputation by half of the minimum value in the dataset.

#### Data Pre-Treatment post-imputation - All of these files were obtained always after the Missing Value Imputation (obtaining the MetAna_Imputed.csv file):

- MetAna_Norm.csv - Normalization by a reference feature ('301/2791.68' feature - random choice) was applied.
- MetAna_QuantileN.csv - Quantile Normalization was applied.
- MetAna_NTotalInt.csv - Normalization by the Total Area in the dataset (Sum) was applied.
- MetAna_PQN.csv - PQN Normalization was applied.


- MetAna_Glog.csv - Generalized Logarithmic Transformation was applied.
- MetAna_Pareto.csv - Pareto Scaling was applied.
- MetAna_AutoScal.csv - Auto Scaling was applied.
- MetAna_RangeScal.csv - Range Scaling was applied.


- MetAna_np.csv - Normalization by a reference feature ('301/2791.68' feature - random choice) and Pareto Scaling was applied.
- MetAna_ng.csv - Normalization by a reference feature ('301/2791.68') and Generalized Logarithmic Transformation was applied.
- MetAna_gp.csv - Generalized Logarithmic transformation and Pareto Scaling was applied.
- MetAna_ngp.csv - Normalization by a reference feature ('301/2791.68'), Generalized Logarithmic Transformation and Pareto Scaling was applied.


##### Note: Missing Value Imputation by half of the minimum value in the dataset is no longer available at MetaboAnalyst 4.0, so the exact analysis to obtain this files can't be replicated there.

MetaboAnalyst 4.0 reference: Chong J, Wishart DS, Xia J. Using MetaboAnalyst 4.0 for Comprehensive and Integrative Metabolomics Data Analysis. Curr Protoc Bioinforma. 2019;68(1):e86. doi:https://doi.org/10.1002/cpbi.86

Link to MetaboAnalyst 4.0: https://www.metaboanalyst.ca/home.xhtml