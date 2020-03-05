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

def remove_feat(df, minsample = 0):
    return trans.keep_atleast(df, min_samples=minsample)

### Normalizations
def Norm_Feat(df, Feat_mass, remove=True):
    return trans.normalize_ref_feature(df, Feat_mass, remove=remove)

def Norm_TotalInt (df):
    """Normalization of a dataset by the total value per columns."""
    
    return df/df.sum()


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
def glog(Spectra, lamb=None):
    """Performs Generalized Logarithmic Transformation on a Spectra (same as MetaboAnalyst's transformation).

    df: Pandas DataFrame.
    lamb: scalar, optional (default: minimum value in the data divided by 10); transformation parameter lambda.

    Returns: DataFrame transformed as log2(y + (y**2 + lamb**2)**0.5)."""

    return trans.glog(df, lamb=lamb)

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


def spectra_proc(df, minsample=0, Feat_mass=None, remove=True, do_glog=False, lamb=None, Pareto=True):
    """Performs any combination of Missing Value Imputation, Normalization by a reference feature, Generalized Logarithmic 
    Transformation and Pareto Scaling of the dataset.

       df: Pandas DataFrame. Can contain missing values.
       minsample: scalar, optional; number between 0 and 1,
       minsample*100 represents the minimum % of samples
       where the feature must be present in order to not be removed.
       
       Feat_mass: index label; m/z of the reference feature to normalize the sample.
       None - Normalization is not performed.
       remove: bool (default: True); True to remove reference feature from data after normalization.
       
       do_glog: bool; Perform or not glog transformation
       lamb: scalar ; transformation parameter for glog

       Pareto: bool (default - True); if True performs Pareto Scaling.

       Returns: Processed DataFrame.
    """
    if minsample != 100: #Missing Value Imputation
        df = NaN_Imputation(df, minsample)

    if Feat_mass is not None: #Normalization by a reference feature
        df = Norm_Feat(df, Feat_mass, remove=remove)

    if do_glog: #glog transformation
        df = glog(df, lamb)

    if Pareto: #Pareto Scaling
        df = ParetoScal(df)
    return df

def dist_discrim(df, Z, method='average'):
    """Give a measure of the normalized distance that a group of samples (same label) is from all other samples in hierarchical clustering.

        This function calculates the distance from a certain number of samples with the same label to the closest samples using the 
        hierarchical clustering linkage matrix and the labels (in Spectra) of each sample. For each set of samples with the same label, it 
        calculates the difference of distances between where the cluster with all the set of samples was formed and the cluster that joins 
        those set of samples with another samples. The normalization of this distance is made by dividing said difference by the max 
        distance between any two cluster. If the samples with the same label aren't all in the same cluster before any other sample joins 
        them, the distance given to this set of samples is zero. It returns the measure of the normalized distance as well as a dictionary 
        with all the calculated distances for all set of samples (labels).

        df: Pandas DataFrame.
        Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage)
        method: str; Available methods - "average", "median". This is the method to give the normalized discrimination distance measure
        based on the distances calculated for each set of samples.

        Returns: (global_distance, discrimination_distances)
        global_distance: float or None; normalized discrimination distance measure
        discrimination_distances: dict: dictionary with the discrimination distance for each label.
    """

    # From linkage table, create dictionaries with the clusters formed at iteration r
    # and the distance between the elements of cluster. See scipy linkage() documentation
    nZ = len(Z)

    dists = {}
    clust = {i: (int(i),) for i in range(0, nZ + 1)}
    for r in range(nZ):
        c1, c2, d, _ = Z[r, :] # unpacking line of iteration r in Z
        clust[nZ + 1 + r] = clust[c1] + clust[c2] # this is addition of tuples
        dists[nZ + 1 + r] = d

    # print('clust ----------------')
    # for c in clust:
    #     print(f'{c} --> {clust[c]}')
    # print('dists ----------------')
    # for d in dists:
    #     print(f'{d} --> {dists[d]}')

    # Get metadata from df
    unique_labels = df.ms.unique_labels
    n_unique_labels = df.ms.label_count
    all_labels = list(df.ms.labels)

    # Create dictionary with number of samples per label
    sample_number = {label: len(df.ms.samples_of(label)) for label in unique_labels}
    min_len = min(sample_number.values())
    max_len = max(sample_number.values())

    # Calculate discriminating distances of a set of samples with the same label
    # store in dictionary `discrims`.
    # `total` accumulates total.

    total = 0
    discrims = {label: 0.0 for label in unique_labels}

    for i in clust:
        len_cluster = len(clust[i])
        # skip if cluster too short or too long
        if not (min_len <= len_cluster <= max_len):
            continue

        labelset = [all_labels[loc] for loc in clust[i]]
        # get first element
        label0 = labelset[0]

        # check if cluster length == exactely the number of samples of label of 1st element.
        # If so, all labels must also be equal
        if len_cluster != sample_number[label0] or labelset.count(label0) != len_cluster:
            continue

        # Compute distances.
        itera = np.where(Z == i)[0][0]
        dif_dist = Z[itera, 2]
        discrims[label0] = (dif_dist-dists[i])/Z[-1, 2]
        # Z[-1,2] - maximum distance between 2 clusters.
        total += discrims[label0]
        # print(f'\n-----------\ncluster {i}, label set ----> {labelset}')
        # print('discrims ---->', discrims)
        # print('separaT ---->', total)

    # Method to quantify a measure of a global discriminating distance for a linkage matrix.
    if method == 'average':
        separaM = total / n_unique_labels
    elif method == 'median':
        separaM = np.median(list(discrims.values()))
        if separaM == 0:
            separaM = None
    else:
        raise ValueError('Method should be one of: ["average", "median"].')
    # print('return values *********')
    # print(separaM)
    # print(discrims)
    # print('************************')

    return separaM, discrims
