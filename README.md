# similarity_share

Shared private repo to work on similarity measures in peak lists of metabolomics data.

This repository contains all the datasets, data analysis, figures and information for tables present in the Master's Dissertation in Biochemistry "Binary Similarity Measures and Mass-Difference Network Analysis as Effective Tools in Metabolomics Data Analysis" of Francisco Traquete developed during this past year (2019/2020) at the Fourier-Transform Ion-Cyclotron-Resonance and Structural Mass Spectrometry (FT-ICR-MS-Lisboa) Laboratory group.

Furthermore, it also contains some of the peripheral work done during this year that helped the dissertation and was ultimately not included in the final version of the dissertation.

Here, we will present how this repository is organized and what is present in each file. Most files are: data, jupyter notebooks or small Python scripts auxiliary to notebooks.

# Organization of the repository - File by file

We will start by presenting the most important files for the dissertation. First, files that were used directly for the analysis and figure (and table) construction: notebooks that perform BinSim and traditional pre-treatments analysis and Sample_MDiNs pre-treatments analysis - main data treatments analysed and compared in the dissertation - and the data files needed to run these notebooks. Then, we will present the other analyses that were performed and not included in the final dissertation, followed by the other BinSim analyses on different datasets (either datasets with different pre-processing or completely different) that weren't included in the final work and, finally, miscellaneous files.

## Files directly used for Figure and Table construction in the dissertation and data files needed to run these:

The aim of the dissertation was to develop two new approaches for the computational analysis of metabolomics data, in the context of profiling and discrimination of biological samples: Binary Similarity (BinSim) and Sample Mass-Difference Networks (Sample MDiNs).

A systematic evaluation of the performance of different multivariate statistical methods in (specifically) discriminating samples of selected high-resolution MS datasets into their respective biological groups when said datasets are treated in one of previously mentioned ways when compared to more established and traditional pre-treatments was made.

The main datasets used for this analyses were called the Negative and Positive Grapevine Datasets (specific peak filtering and peak alignment - explained in the notebooks) and the Yeast and Yeast Formula Datasets. These datasets pre-processed in other ways as well as another set of datasets called the MetaboLights dataset were also analysed in other notebooks - see 'Other BinSim analyses notebooks').

### BinSim (and traditional pre-treatments) analysis of the Grapevine and Yeast Datasets

These notebooks perform data analysis thorugh different multivariate statistical methods of specific datasets (presented in the dissertation) after specific data pre-treatments: Binary Similarity (BinSim) or some combinations of more established and traditional pre-treatments (compare the results between these).

- BinSim_Analysis_GD11_all2_groups2all1.ipynb - (BinSim and traditional treatments) Data Analysis of the Negative and Positive Grapevine Datasets.
- BinSim_Analysis_YD_notnorm.ipynb - (BinSim and traditional treatments) Data Analysis of the Yeast and Yeast Formula Datasets.

Information for Fig 3.1 to Fig 3.6, Suppl. Fig. 6.1 to Suppl. Fig. 6.8, Table 3.1 to Table 3.3, Suppl. Table 6.1 and 6.2. Also, base for Fig. 1.1, Fig. 1.3, Fig. 2.2.

Note: Ahead of the GD11 and YD name of the files, it is an indication of the pre-processing done to the datasets that will be analysed. Explanation of the pre-processing is done in the notebooks (first file has 2 different pre-processed set of datasets - 'all2' represents the one used in the dissertation). The analysis done in these notebooks is replicated for the datasets pre-processed in different ways - see 'Other BinSim analysis on Grapevine and Yeast datasets' section.

#### Data Files and Dataset Construction

##### Obtaining the Grapevine Datasets

The Negative and Positive Grapevine Datasets were obtained with the peak_alignment.ipynb from each grapevine sample information in the 'data' folder. The different peak alignments and peak filterings done using the `align` function and stores in hdf stores. The different ways alignment and filtering were performed is explained in peak_alignment.ipynb as well as the nomenclature given to each alignment and filtering.

- peak_alignment.ipynb - aligns and filters (based on two parameters) the grapevine samples using the `align` function from metabolinks, explains the alignments and filterings as well as the nomenclature given.
- 'data' folder (not currently in repository) - has all the mass lists, intensity and metadata for each grapevine sample (in negative and positive modes).

Stores:
- alignments_new.h5 - has the datasets used in the dissertation - and alignments_old.h5 (new compressed alignments - updated `align` function). They differ on the linkage method used in the updated `align` function ('hc' median and complete, respectively) of metabolinks.
- alignments.h5 - alignments made with the old `align` function from metabolinks.

Note:
- generate_alignments.py - similar to what the peak_alignment.ipynb notebook does.
- comparealignments.py - compare alignments made in the 3 different hdf stores.
- merge_near.py - merges close features - not needed anymore.

##### Yeast Datasets

The Yeast Datasets were obtained from the MetaboScape 4.0 software (Bruker Daltonics). The main difference between the files is that one was not normalized by Metaboscape (used in the dissertation) and the other was. Other parameters used are described in the respective notebooks.

- 5yeasts_notnorm.csv - Yeast Dataset not normalized by MetaboScape.
- 5yeasts_norm.csv - Yeast Dataset normalized by MetaboScape (see use in the 'Other BinSim analysis on Grapevine and Yeast datasets' section - BinSim_YD_norm.ipynb).

### Sample MDiNs as a Data Pre-Treatment - Data analysis of the Grapevine and Yeast Datasets

These notebooks perform the same data analysis (multivariate statistical methods) of specific datasets (presented in the dissertation) as the BinSim section but treating these datasets by building sample MDiNs and analysing them by different Network Analysis methods (results from the different analysis methods are compared).

- Sample_MDiNs_Yeast.ipynb - Data Analysis of the Sample MDiNs made from the Yeast Dataset.
- Sample_MDiNs_Grapevine.ipynb - Data Analysis of the Sample MDiNs made from the Negative and Positive Grapevine Datasets.

Information for Fig 3.7 to Fig 3.10, Suppl. Fig. 6.9 to Suppl. Fig. 6.12, Table 2.2, Table 3.4 to Table 3.7, Suppl. Table 6.3.

#### Mass-Difference Networks (MDiNs)

Full MDiNs were built with MetaNetter 2.0 plugin of Cytoscape using either the transformation list (TF) made and discussed in Sample_MDiNs_Yeast.ipynb or the transformation list of MetaNetter (MN) under same restrictions as explained in the same notebook. The files that contain the networks are:

- Net_YD_TF.graphml - Full Yeast dataset MDiN built with our transformation list.
- Net_YD_MN.graphml - Full Yeast dataset MDiN built with MetaNetter's transformation list.
- Net_YD_BY0_1.graphml - Sample MDiN of replicate nº 1 of strain BY4741 of the Yeast Dataset built with our transformation list.
- Net_NGD_TF.graphml - Full Negative Grapevine dataset MDiN built with our transformation list.
- Net_NGD_MN.graphml - Full Negative Grapevine dataset MDiN built with MetaNetter's transformation list.
- Net_PGD_TF.graphml - Full Positive Grapevine dataset MDiN built with our transformation list.
- Net_PGD_MN.graphml - Full Positive Grapevine dataset MDiN built with MetaNetter's transformation list.

### Python files with functions useful for data analysis of the notebooks

- scaling.py - has multiple functions for application of different traditional data pre-treatments (some of them were moved to metabolinks Python package).
- multianalysis.py - has multiple functions for the application of multivariate analysis methods (from scikit-learn Python package) in the specific ways we want and extracting the results we want. Development of some evaluation metrics for clustering techniques.


- test_scaling.py - tests to observe if the treatments are being correctly applied by comparing with the treatments made in MetaboAnalyst software based on an example file from this software.
- test_multianalysis.py - very few tests to see if some of the developed metrics are obtaining the results we want (scarce tests).

Note:
- MetAnalyst folder - has an example dataset file present in MetaboAnalyst and multiple files of this dataset after specific pre-treatments (to compare their application of pre-treatments with the one in scaling.py).
- MetAnalyst_Example.ipynb - a few more tests to observe if the treatments are being correctly applied like test_scaling.py.

### Other Notebook that has information for figure in the dissertation

- VanKrevelen_Series_analysis.ipynb - Has the examples in Fig. 1.4. Also uses the points in Van Krevelen diagram (('H/C','O/C') pairs) and the number of species in the different composition series ('CHO', 'CHOS', 'CHON', 'CHNS', 'CHONS', 'CHOP', 'CHONP','CHONSP') as a treatment in the data for discrimination of samples. This uses the following file:
- ThesisSupportExcel.xlsx - excel file with the data from the Yeast Dataset made with slightly different parameters and different Formula assignment - this was just used for the example figure in the dissertation.

## Other analyses made besides BinSim and Sample MDiNs

- Sample_Formula_Networks.ipynb - Similar analysis to Sample_MDiNs_Yeast.ipynb, however the networks are built with the formulas already assigned to the Yeast Dataset by observing the differences in the formulas itself to establish connections (different approach).


- FormGeneration_Assignment.ipynb - This notebook contains an algorithm to build a database of possible formulas in a specific m/z interval (according to some of the parameters that can be changed) and an algorithm to assign formulas from this database to a list of neutral masses (also parameters than can be changed - 3 different variants of the algorithm) that are then compared to the Formulas assignment made in a freely available software 'Formularity' (available here: https://omics.pnl.gov/software/formularity). The files used were neutral mass lists as obtained from the Formularity software (after formula assignment) to facilitate formula assignment comparison. These were in the following folder:
- 'Formularity_data' folder - folder with multiple files (use of different parameters in the Assignment by Formularity) obtained from the Formularity software either from replicate nº 1 of BY4741 strain of the Yeast dataset or the example file provided by Formularity.


- Form_ratio_test(Needs files to run).ipynb - this is an auxiliary notebook to FormGeneration_Assignment.ipynb to observe the most common ratios (certain element number of atoms to carbon atoms in a formula/metabolite) in a database to add as an additional criterion (stricter elemental ratios) in the formula assignment algorithm. The database used was ChEBI (https://www.ebi.ac.uk/chebi/). As the name suggests, this notebook doesn't run without the specific files from CHEBI (explained in the notebook), which aren't present in the repository due to their size.

## Other BinSim analyses notebooks

### Other BinSim analyses on Grapevine and Yeast datasets (different pre-processings)

The other alignments of the Negative and Positive Grapevine Datasets in the hdf stores were analysed in these notebooks and the Yeast Dataset normalized MetaboScape also was. Furthermore, one of these notebooks marked as old is an older version of another of the notebooks, since the latter (that is more recent) used to analyse datasets that are no longer present in the repository.

##### This analysis was very similar to BinSim_Analysis_GD11_all2_groups2all1.ipynb and BinSim_Analysis_YD_notnorm.ipynb that are more well organized and commented to follow the analysis. Some parameters in the latter were improved to ameliorate the figures and the analysis which weren't applied in these other notebooks. Thus, we recommend to see the better commented and organized notebooks first. These notebooks don't have the 'Analysis' on the name to separate them.

- BinSim_GD11_g2all1_groups3all1.ipynb

- BinSim_GD14_all6_all13.ipynb
- BinSim_GD14_all6_all13_old.ipynb
- BinSim_GD14_original_g2all1_groups3all1.ipynb


- BinSim_YD_norm.ipynb

### Other BinSim analysis on specific MetaboLights datasets

To further corroborate the results, the same analysis was also done on online available datasets. These datasets were taken from the MetaboLights (https://www.ebi.ac.uk/metabolights/) repository (each notebook describes from where each dataset was obtained). These results corroborated the conclusions of the dissertation but were not included in the final text to avoid repetition. These datasets were chosen since they had multiple groups and a decent amount of samples while having features with missing values and being somewhat well described.

- BinSim_ML_Bluebells.ipynb
- BinSim_ML_FC.ipynb
- BinSim_ML_Wolbachia.ipynb

##### Data files:
- 'Metabolights' folder - has the data of the MetaboLights datasets for the previous notebooks.

## Miscellaneous

- graphlets.png - image from the orbits and different graphlets up to 5-nodes.
- figs_paper_vitis.ipynb - unrelated notebook.
- tabela5yeasts11-3-2020.xlsx - excel file with the data from the Yeast Dataset made with different parameters and very different Formula assignment parameters than the other datasets - no longer used.
