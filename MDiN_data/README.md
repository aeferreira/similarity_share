# MDiN data

This folder contains the MDiNs that were built with MetaNetter 2.0 plugin of Cytoscape in graphml format. These were made from the mass lists of the Yeast Dataset ('5yeasts_notnorm.csv'), Negative and Positive Grapevine Datasets ('all_1ppm_min2_neg' and 'all_1ppm_min2_pos' of the 'alignments_new.h5', respectively) after treatment to transform the m/z peaks into "neutral" masses (see Sample_MDiNs_Yeast.ipynb and Sample_MDiNs_Grapevine.ipynb). The parameters used were 1 ppm and the transformation list used were either our transformation list (TF) made and discussed in Sample_MDiNs_Yeast.ipynb or the transformation list of MetaNetter (MN) under some restrictions (only transformations until 80 Da and without Arginine, Arg to Ornitine and pyrophosphate transformations). 

The nomenclature for the files is:

- Net_Dataset_TransformationList.graphml

Thus, the files that contain the networks are:

- Net_YD_TF.graphml - Full Yeast dataset MDiN built with our transformation list.
- Net_YD_MN.graphml - Full Yeast dataset MDiN built with MetaNetter's transformation list.
- Net_YD_BY0_1.graphml - Sample MDiN of replicate nº 1 of strain BY4741 of the Yeast Dataset built with our transformation list.
- Net_NGD_TF.graphml - Full Negative Grapevine dataset MDiN built with our transformation list.
- Net_NGD_MN.graphml - Full Negative Grapevine dataset MDiN built with MetaNetter's transformation list.
- Net_PGD_TF.graphml - Full Positive Grapevine dataset MDiN built with our transformation list.
- Net_PGD_MN.graphml - Full Positive Grapevine dataset MDiN built with MetaNetter's transformation list.