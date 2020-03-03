import pandas as pd
import metabolinks as mtl
import metabolinks.transformations as trans
import numpy as np

"""Elimination of features with too many missing values, missing value estimation, normalization methods, generalized logarithmic
transformation, reference feature estimation and centering and scaling methods. Discrimination distance of hierarchical clustering."""


### Missing Value Imputation and feature removal

def NaN_Imputation(df, minsample=0):
    df = trans.keep_atleast(df, min_samples=minsample)
    df = trans.fillna_frac_min(df, fraction=0.5)
    return df


# def NaN_Imputation(Spectra, minsample = 0):
#     """Remove features with a certain % of missing values, replaces remaining ones by half of the minimum value of the original data.

#        Spectra: AlignedSpectra object (from metabolinks).
#        minsample: scalar (default: 0); minimum percentage of samples a feature must be to not be removed.

#        Returns: AlignedSpectra object (from metabolinks); Equal to Spectra but with some features removed and missing values replaced.
#     """

#     Imputated = Spectra
#     df = Imputated.data
#     if minsample != 0:
#         NumValues = Imputated.data.notnull()
#         a = 0
#         for i in range(0, len(NumValues)):
#             if sum(NumValues.iloc[i, :]) < minsample*Imputated.sample_count:
#                 # Taking away features that appear in less of minsample% of samples.
#                 df = df.drop([df.iloc[a].name])
#             else:
#                 a = a + 1

#     Imputated = AlignedSpectra(
#         df, sample_names = Imputated.sample_names, labels = Imputated.labels)
#     # Replace missing values
#     Imputated.data.fillna(min(Imputated.data.min()/2), inplace = True)

#     return Imputated

def remove_feat(df, minsample = 0):
    return trans.keep_atleast(df, min_samples=minsample)


### Normalizations
def Norm_Feat(df, Feat_mass, remove=True):
    return trans.normalize_ref_feature(df, Feat_mass, remove=remove)

# def Norm_Feat(Spectra, Feat_mass, remove=True):
#     """Normalization of a dataset by a reference feature.

#        Spectra: AlignedSpectra object (from metabolinks).
#        Feat_mass: scalar; m/z of the reference feature to normalize the sample.
#        remove: bool; True to remove reference feature from data after normalization.

#        Returns: AlignedSpectra object (from metabolinks); normalized spectra.
#        """

#     temp = Spectra.data.copy()
#     for i in range(Spectra.sample_count):
#         temp.iloc[:, i] = temp.iloc[:, i]/Spectra.data.loc[Feat_mass][i]
#     if remove:
#         temp = temp.drop([Feat_mass])

#     return AlignedSpectra(temp, sample_names=Spectra.sample_names, labels=Spectra.labels)


def Norm_TotalInt (Spectra):
    """Normalization of a dataset by the total peak intensity in each Spectra.

       Spectra: AlignedSpectra object (from metabolinks).

       Returns: AlignedSpectra object (from metabolinks); normalized spectra.
    """
    return AlignedSpectra(Spectra.data/Spectra.data.sum(), sample_names=Spectra.sample_names, labels = Spectra.labels)


#Needs double-checking
def Norm_PQN(Spectra, ref_sample = 'mean'):
    """Normalization of a dataset by the Probabilistic Quotient Normalization method.

       Spectra: AlignedSpectra object (from metabolinks).
       ref_sample: reference sample to use in PQ Normalization, types accepted: "mean" (default, reference sample will be the intensity
    mean of all samples for each feature - useful for when there are a lot of imputed missing values), "median" (reference sample will
    be the intensity median of all samples for each feature - useful for when there aren't a lot of imputed missing values), sample name
    (reference sample will be the sample with the given name in the dataset) or list with the intensities of all peaks that will
    directly be the reference sample (pandas Series format not accepted - list(Series) is accepted).

       Returns: AlignedSpectra object (from metabolinks); normalized spectra.
    """
    #"Building" the reference sample based on the input given
    if ref_sample == 'mean': #Mean spectre of all samples
        ref_sample2 = Spectra.data.T/Spectra.data.mean(axis = 1)
    elif ref_sample == 'median': #Median spectre of all samples
        ref_sample2 = Spectra.data.T/Spectra.data.median(axis = 1)
    elif ref_sample in Spectra.sample_names: #Specified sample of the spectra
        ref_sample2 = Spectra.data.T/Spectra.data.loc[:,ref_sample]
    else: #Actual sample given
        ref_sample2 = Spectra.data.T/ref_sample
    #Normalization Factor and Normalization
    Norm_fact = ref_sample2.median(axis = 1)
    return AlignedSpectra(Spectra.data/Norm_fact, sample_names = Spectra.sample_names, labels = Spectra.labels)


def Norm_Quantile(Spectra, ref_type = 'mean'):
    """Normalization of a dataset by the Quantile Normalization method.

       Missing Values are temporarily replaced with 0 (and count as 0) until normalization is done. Quantile Normalization is more
    useful with no/low number of missing values.

       Spectra: AlignedSpectra object (from metabolinks).
       ref_type: str (default: 'mean'); reference sample to use in Quantile Normalization, types accepted: 'mean' (default,
    reference sample will be the means of the intensities of each rank), 'median' (reference sample will be the medians of the
    intensities for each rank).

       Returns: AlignedSpectra object (from metabolinks); normalized spectra.
    """
    #Setting up the temporary dataset with missing values replaced by zero and dataframes for the results
    norm = Spectra.data.copy().replace({np.nan:0})
    ref_spectra = Spectra.data.copy()
    ranks = Spectra.data.copy()

    for i in range(len(norm)):
        #Determining the ranks of each entry in the same row (same variable) in the dataset
        ref_spectra.iloc[i] = norm.iloc[i].sort_values().values
        ranks.iloc[i] = norm.iloc[i].rank(na_option = 'top')

    #Determining the reference sample for normalization based on the ref_type chosen (applied on the columns).
    if ref_type == 'mean':
        ref_sample = ref_spectra.mean(axis = 0).values
    elif ref_type == 'median':
        ref_sample = ref_spectra.median(axis = 0).values
    else:
        raise ValueError('Type not recognized. Available ref_type: "mean", "median".')

    #Replacing the values in the dataset for the reference sample values based on the rankscalculated  earlier for each entry
    for i in range(len(ranks)):
        for j in range(len(ranks.columns)):
            if ranks.iloc[i,j] == round(ranks.iloc[i,j]):
                norm.iloc[i,j] = ref_sample[int(ranks.iloc[i,j])-1]
            else: #in case the rank isn't an integer and ends in .5 (happens when a pair number of samples have the same
                  #value in the same column - after ordering from lowest to highest values by row).
                norm.iloc[i,j] = np.mean((ref_sample[int(ranks.iloc[i,j]-1.5)], ref_sample[int(ranks.iloc[i,j]-0.5)]))

    #Replacing 0's by missing values and creating the AlignedSpectra object for the output
    return AlignedSpectra(norm.replace({0:np.nan}), sample_names = Spectra.sample_names, labels = Spectra.labels)


### Transformations
def glog(Spectra, lamb = True):
    """Performs Generalized Logarithmic Transformation on a Spectra (same as MetaboAnalyst's transformation).

       Spectra: AlignedSpectra object (from metabolinks).
       lamb: scalar, optional (default: True);  transformation parameter, if True lamb = minimum value in the data divided by 10.

       Returns: AlignedSpectra object (from metabolinks); transformed spectra by a factor of log(y + (y**2 + lamb**2)**0.5).
       """
    #Defining lambda
    if lamb == True:
        lamb = min(Spectra.data.min()/10)
    #Applying the equation
    y = Spectra.data.copy()
    y = np.log2((y + (y**2 + lamb**2)**0.5)/2)
    return AlignedSpectra(y, sample_names = Spectra.sample_names, labels = Spectra.labels)


### Centering and Scalings (acomodates Missing Values)

def ParetoScal(df):
    return trans.pareto_scale(df)

def MeanCentering(Spectra):
    """Performs Mean Centering on an AlignedSpectra object.

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.

       Returns: Aligned Spectra object (from metabolinks); Mean Centered Spectra."""

    return AlignedSpectra((Spectra.data.T - Spectra.data.mean(axis = 1)).T, sample_names = Spectra.sample_names, labels = Spectra.labels)


def AutoScal(Spectra):
    """Performs Autoscaling on an AlignedSpectra object.

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.

       Returns: Aligned Spectra object (from metabolinks); Auto Scaled Spectra."""

    scaled_aligned = Spectra.data.copy()
    std = Spectra.data.std(axis = 1)
    # Applying Autoscaling
    scaled_aligned = ((Spectra.data.T - Spectra.data.mean(axis = 1))/std).T

    # Return scaled spectra
    return AlignedSpectra(scaled_aligned, sample_names = Spectra.sample_names, labels = Spectra.labels)


def RangeScal(Spectra):
    """Performs Range Scaling on an AlignedSpectra object.

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.

       Returns: Aligned Spectra object (from metabolinks); Ranged Scaled Spectra."""

    scaled_aligned = Spectra.data.copy()
    ranges = Spectra.data.max(axis = 1) - Spectra.data.min(axis = 1) # Defining range for every feature
    # Applying Range scaling to each feature
    for j in range(0, len(scaled_aligned)):
        if ranges.iloc[j] == 0: # No difference between max and min values
            scaled_aligned.iloc[j, :] = Spectra.data.iloc[j, ]
        else:
            scaled_aligned.iloc[j, :] = (Spectra.data.iloc[j, ] - Spectra.data.iloc[j, ].mean())/ranges.iloc[j]

    # Return scaled spectra
    return AlignedSpectra(scaled_aligned, sample_names = Spectra.sample_names, labels = Spectra.labels)


def VastScal(Spectra):
    """Performs Vast Scaling on an AlignedSpectra object.

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.

       Returns: Aligned Spectra object (from metabolinks); Vast Scaled Spectra."""

    scaled_aligned = Spectra.data.copy()
    std = Spectra.data.std(axis = 1)
    mean = Spectra.data.mean(axis = 1)
    # Applying Vast Scaling to each feature
    scaled_aligned = (((Spectra.data.T - mean)/std)/(mean/std)).T

    # Return scaled spectra
    return AlignedSpectra(scaled_aligned, sample_names = Spectra.sample_names, labels = Spectra.labels)


def LevelScal(Spectra, average = True):
    """Performs Level Scaling on an AlignedSpectra object. (See van den Berg et al., 2006).

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.
       average: bool (Default - True); if True mean-centered data is divided by the mean spectra, if False it is divided by the median
    spectra.

       Returns: Aligned Spectra object (from metabolinks); Level Scaled Spectra."""

    mean = Spectra.data.mean(axis = 1)
    # Applying Level Scaling to each feature
    if average == True:
        scaled_aligned = ((Spectra.data.T - mean)/mean).T
    elif average == False:
        scaled_aligned = ((Spectra.data.T - mean)/Spectra.data.median(axis = 1)).T

    # Return scaled spectra
    return AlignedSpectra(scaled_aligned, sample_names = Spectra.sample_names, labels = Spectra.labels)


### Miscellaneous
def search_for_ref_feat(df, approx_mass):
    """Estimates a peak m/z to be the reference feature. 

       df: DataFrame.
       approx_mass: scalar, approximate mass of the reference feature to search for.

       Return: scalar, scalar; peak m/z that is estimated to belong to the reference feature (must be present in all samples,
    must be at a max length of 1 m/z of the approximate mass given) - the scalar returned is the closest peak that fulfills 
    these two conditios; m/z difference of approximate m/z and estimated m/z.
    """
    # Condition 1: Be at a max length of 1 of the approximate mass given.
    rest1 = df.index[df.index < approx_mass + 1]
    rest2 = rest1[rest1 > approx_mass-1]

    # Condition 2: Be present in every sample.
    feat_est = []
    for i in range(len(rest2)):
        if sum(np.isnan(df.loc[rest2[i]])) == 0:
            feat_est.append(rest2[i])

    if len(feat_est) == 1:
        return feat_est[0], abs(feat_est[0] - approx_mass)

    elif len(feat_est) == 0:
        raise ValueError ('No feature is present in all sample around approx_mass.')
        # return print('Error - No feature is present in all sample around approx_mass')

    # Tiebraker: Be closer to the approximate mass (m/z) given than other peaks that fulfill previous conditions.
    else:
        mass_diff = []
        for i in range(len(feat_est)):
            mass_diff.append(abs(feat_est[i]-approx_mass))

        return feat_est[mass_diff.index(min(mass_diff))], min(mass_diff)


def spectra_proc(Spectra, minsample=0, Feat_mass=False, remove=True, lamb= 'False', Pareto = True):
    """Performs any combination of Missing Value Imputation, Normalization by a reference feature, Generalized Logarithmic 
    Transformation and Pareto Scaling of the dataset.

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.
       minsample: scalar, optional; number between 0 and 1, minsample*100 represents the minimum % of samples where the feature must 
    be present in order to not be removed.
       Feat_mass: scalar (default: False); m/z of the reference feature to normalize the sample. False - Normalization is not performed.
       remove: bool (deafult: True); True to remove reference feature from data after normalization.
       lamb: scalar (default - 'False');  transformation parameter, if 'False', glog transformation is not performed.
       Pareto: bool (default - True); if True performs Pareto Scaling.

       Returns: Processed Aligned Spectra object (from metabolinks).
    """
    if minsample != 100: #Missing Value Imputation
        Spectra = NaN_Imputation(Spectra, minsample)

    if Feat_mass != False: #Normalization by a reference feature
        Spectra = Norm_Feat(Spectra, Feat_mass, remove = remove)

    if lamb != 'False': #glog transformation
        Spectra = glog(Spectra, lamb)

    if Pareto != False: #Pareto Scaling
        Spectra = ParetoScal(Spectra)
    return Spectra

def dist_discrim(Spectra, Z, method='average'):
    """Gives a measure of the normalized distance that a group of samples (same label) is from all other samples in hierarchical clustering.

       This function calculates the distance from a certain number of samples with the same label to the closest samples using the 
    hierarchical clustering linkage matrix and the labels (in Spectra) of each sample. For each set of samples with the same label, it 
    calculates the difference of distances between where the cluster with all the set of samples was formed and the cluster that joins 
    those set of samples with another samples. The normalization of this distance is made by dividing said difference by the max 
    distance between any two cluster. If the samples with the same label aren't all in the same cluster before any other sample joins 
    them, the distance given to this set of samples is zero. It returns the measure of the normalized distance as well as a dictionary 
    with all the calculated distances for all set of samples (labels).

       Spectra: AlignedSpectra object (from metabolinks).
       Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage)
       method: str; Available methods - "average", "median". This is the method to give the normalized discrimination distance measure
    based on the distances calculated for each set of samples.

       Returns: (scalar, dictionary); normalized discrimination distance measure, dictionary with the discrimination distance for each
    set of samples.
    """

    # Create dictionaries with the clusters formed at iteration r
    # and the distance between the elements of cluster.
    dists = {}
    clust = {}
    for i in range(0, len(Z)+1):
        clust[i] = (float(i),)
    for r in range(0, len(Z)):
        clust[len(Z)+1+r] = clust[Z[r, 0]] + clust[Z[r, 1]]
        dists[len(Z)+1+r] = Z[r, 2]

    unique_labels = Spectra.ms.unique_labels
    n_unique_labels = Spectra.ms.label_count
    all_labels = list(Spectra.ms.labels)

    #Create dictionary with number of samples for each group
    sample_number = {label: len(Spectra.ms.samples_of(label)) for label in unique_labels}

    # Calculate discriminating distances of a set of samples with the same label
    # and storing in a dictionary.

    separaT = 0
    separa = {label: [0] * n_unique_labels for label in unique_labels}

    for i in clust:
        labelset = [all_labels[int(clust[i][j])] for j in range(len(clust[i]))]

        #check if cluster length = the number of samples of the group of one of the samples that belong to the cluster.
        if len(clust[i]) == sample_number[labelset[0]]:
            #labelset = [Spectra.labels[int(clust[i][j])] for j in range(sample_number)]
            # All labels must be the same.
            if labelset.count(labelset[0]) == len(labelset):
                itera = np.where(Z == i)[0][0]
                dif_dist = Z[itera, 2]
                separa[labelset[0]] = (dif_dist-dists[i])/Z[-1, 2]#Z[-1,2] - maximum distance between 2 clusters.
                separaT = separaT + separa[labelset[0]]

    # Method to quantify a measure of a global discriminating distance for a linkage matrix.
    if method == 'average':
        separaM = separaT/n_unique_labels
    elif method == 'median':
        separaM = np.median(list(separa.values()))
        if separaM == 0:
            separaM = None
    else:
        raise ValueError(
            'Method not recognized. Available methods: "average", "median".')

    return separaM, separa

def dist_discrim_new(df, Z, method='average'):
    """Gives a measure of the normalized distance that a group of samples (same label) is from all other samples in hierarchical clustering.

       This function calculates the distance from a certain number of samples with the same label to the closest samples using the 
    hierarchical clustering linkage matrix and the labels (in Spectra) of each sample. For each set of samples with the same label, it 
    calculates the difference of distances between where the cluster with all the set of samples was formed and the cluster that joins 
    those set of samples with another samples. The normalization of this distance is made by dividing said difference by the max 
    distance between any two cluster. If the samples with the same label aren't all in the same cluster before any other sample joins 
    them, the distance given to this set of samples is zero. It returns the measure of the normalized distance as well as a dictionary 
    with all the calculated distances for all set of samples (labels).

       Spectra: DataFrame.
       Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage)
       method: str; Available methods - "average", "median". This is the method to give the normalized discrimination distance measure
    based on the distances calculated for each set of samples.

       Returns: (scalar, dictionary); normalized discrimination distance measure, dictionary with the discrimination distance for each
    set of samples.
    """

    # Creating dictionaries with the clusters formed at iteration r and the distance between the elements of said cluster.
    dists = {}
    clust = {}
    nZ = len(Z)
    for i in range(0, nZ+1):
        clust[i] = (float(i),)
    for r in range(0, nZ):
        clust[nZ + 1 + r] = clust[Z[r, 0]] + clust[Z[r, 1]]
        dists[nZ + 1 + r] = Z[r, 2]

    #Creating dictionary with number of samples for each group
    unique_labels = list(df.ms.unique_labels)
    n_unique = len(unique_labels)
    labels = list(df.ms.labels)
    
    sample_number = {}
    for label in unique_labels:
        sample_number[label] = len(df.ms.samples_of(label))

    # Calculating discriminating distances of a set of samples with the same label and storing in a dictionary.
    separaT = 0
    separa = {label: [0]*n_unique for label in unique_labels}
    for i in clust:
        label = [labels[int(clust[i][j])] for j in range(len(clust[i]))]
        #check if cluster length = the number of samples of the group of one of the samples that belong to the cluster.
        if len(clust[i]) == sample_number[label[0]]:
            #label = [Spectra.labels[int(clust[i][j])]
                     #for j in range(sample_number)]
            # All labels must be the same.
            if label.count(label[0]) == len(label):
                itera = np.where(Z == i)[0][0]
                dif_dist = Z[itera, 2]
                separa[label[0]] = (dif_dist-dists[i])/Z[-1, 2]#Z[-1,2] - maximum distance between 2 clusters.
                separaT = separaT + separa[label[0]]

    # Method to quantify a measure of a global discriminating distance for a linkage matrix.
    if method == 'average':
        separaM = separaT / n_unique
    elif method == 'median':
        separaM = np.median(list(separa.values()))
        # if separaM == 0:
        #     separaM = None
    else:
        raise ValueError(
            'Method not recognized. Available methods: "average", "median".')

    return separaM, separa
