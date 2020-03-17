import pandas as pd
import metabolinks as mtl
import metabolinks.transformations as trans
import numpy as np

from scipy.cluster import hierarchy as hier

"""Elimination of features with too many missing values, missing value estimation, normalization methods, generalized logarithmic
transformation, reference feature estimation and centering and scaling methods. Discrimination distance of hierarchical clustering."""


### Missing Value Imputation and feature removal

def NaN_Imputation(df, minsample=0):
    df = trans.keep_atleast(df, min_samples=minsample)
    df = trans.fillna_frac_min(df, fraction=0.5)
    return df

# ---------- Normalizations --------------
def Norm_Feat(df, Feat_mass, remove=True):
    return trans.normalize_ref_feature(df, Feat_mass, remove=remove)

def Norm_TotalInt (df):
    """Normalization of a dataset by the total value per columns."""

    return df/df.sum(axis=0)

# Needs double-checking
def Norm_PQN(df, ref_sample='mean'):
    """Normalization of a dataset by the Probabilistic Quotient Normalization method.

       df: Pandas DataFrame.
       ref_sample: reference sample to use in PQ Normalization, types accepted: "mean" (default, reference sample will be the intensity
    mean of all samples for each feature - useful for when there are a lot of imputed missing values), "median" (reference sample will
    be the intensity median of all samples for each feature - useful for when there aren't a lot of imputed missing values), sample name
    (reference sample will be the sample with the given name in the dataset) or list with the intensities of all peaks that will
    directly be the reference sample (pandas Series format not accepted - list(Series) is accepted).

       Returns: Pandas DataFrame; normalized spectra.
    """
    #"Building" the reference sample based on the input given
    if ref_sample == 'mean': #Mean spectre of all samples
        ref_sample2 = df.T / df.mean(axis = 1)
    elif ref_sample == 'median': #Median spectre of all samples
        ref_sample2 = df.T/df.median(axis = 1)
    elif ref_sample in df.cdl.samples: #Specified sample of the spectra
        ref_sample2 = df.T/df.loc[:,ref_sample]
    else: #Actual sample given
        ref_sample2 = df.T/ref_sample
    #Normalization Factor and Normalization
    Norm_fact = ref_sample2.median(axis=1)
    return df / Norm_fact


def Norm_Quantile(df, ref_type='mean'):
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
    norm = df.copy().replace({np.nan:0})
    ref_spectra = df.copy()
    ranks = df.copy()

    for i in range(len(norm)):
        #Determining the ranks of each entry in the same row (same variable) in the dataset
        ref_spectra.iloc[i] = norm.iloc[i].sort_values().values
        ranks.iloc[i] = norm.iloc[i].rank(na_option='top')

    #Determining the reference sample for normalization based on the ref_type chosen (applied on the columns).
    if ref_type == 'mean':
        ref_sample = ref_spectra.mean(axis=0).values
    elif ref_type == 'median':
        ref_sample = ref_spectra.median(axis=0).values
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
    return norm.replace({0:np.nan})


### Transformations
def glog(df, lamb=None):
    """Performs Generalized Logarithmic Transformation on a Spectra (same as MetaboAnalyst's transformation).

    df: Pandas DataFrame.
    lamb: scalar, optional (default: global minimum divided by 10); transformation parameter lambda.

    Returns: DataFrame transformed as log2(y + (y**2 + lamb**2)**0.5)."""

    return trans.glog(df, lamb=lamb)

# Centering and Scalings (acomodates Missing Values)

def ParetoScal(df):
    return trans.pareto_scale(df)

def MeanCentering(df):
    """Performs Mean Centering.

       df: Pandas DataFrame. It can include missing values.

       Returns: DataFrame; Mean Centered Spectra."""
    return df.sub(df.mean(axis=1), axis=0)


def AutoScal(df):
    """Performs Autoscaling on an AlignedSpectra object.

       df: Pandas DataFrame. Can include missing values.

       Returns: Pandas DataFrame; Auto Scaled Spectra.

       This is x -> (x - mean(x)) / std(x) per feature"""

    # TODO: verify if the name of this transformation is "Standard scaling"
    # TODO: most likely it is known by many names (scikit-learn has a SatndardScaler transformer)
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    df2 = df.sub(means, axis=0).div(stds, axis=0)
    return df2

def RangeScal(df):
    """Performs Range Scaling on an AlignedSpectra object.

       df: PAndas DataFrame. It can include missing values.

       Returns: Pandas DataFrame; Ranged Scaled Spectra."""

    scaled_aligned = df.copy()
    ranges = df.max(axis=1) - df.min(axis=1) # Defining range for every feature
    # Applying Range scaling to each feature
    for j in range(0, len(scaled_aligned)):
        if ranges.iloc[j] == 0: # No difference between max and min values
            scaled_aligned.iloc[j, :] = df.iloc[j, ]
        else:
            scaled_aligned.iloc[j, :] = (df.iloc[j, ] - df.iloc[j, ].mean()) / ranges.iloc[j]

    return scaled_aligned


def VastScal(df):
    """Performs Vast Scaling on an AlignedSpectra object.

       df: PAndas DataFrame. It can include missing values.

       Returns: Pandas DataFrame; Vast Scaled Spectra."""

    # scaled_aligned = df.copy()
    std = df.std(axis=1)
    mean = df.mean(axis=1)
    # Applying Vast Scaling to each feature
    scaled_aligned = (((df.T - mean)/std)/(mean/std)).T

    # Return scaled spectra
    return scaled_aligned


def LevelScal(df, average=True):
    """Performs Level Scaling on a DataFrame. (See van den Berg et al., 2006).

    df: Pandas dataFrame. It can include missing values.
    average: bool (Default - True); if True mean-centered data is divided by the mean spectra, if False it is divided by the median
    spectra.

    Returns: Pandas DataFrame; Level Scaled Spectra."""

    mean = df.mean(axis=1)
    # Applying Level Scaling to each feature
    if average == True:
        scaled_aligned = ((df.T - mean)/mean).T
    elif average == False:
        scaled_aligned = ((df.T - mean)/df.median(axis=1)).T
    else:
        raise ValueError ('Average is a boolean argument. Only True or False accepted.')

    # Return scaled spectra
    return scaled_aligned


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
        raise ValueError ('No feature is present in all samples around approx_mass.')
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

    # Get metadata from df
    unique_labels = df.cdl.unique_labels
    n_unique_labels = df.cdl.label_count
    all_labels = list(df.cdl.labels)
    ns = len(df.cdl.samples)

    # Create dictionary with number of samples per label
    sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}
    min_len = min(sample_number.values())
    max_len = max(sample_number.values())

    # to_tree() returns root ClusterNode and ClusterNode list
    _, cn_list = hier.to_tree(Z, rd=True)

    # print('results from to_tree ----------------')
    # for cn in cn_list[ns:]:
    #     n_in_cluster = cn.get_count()
    #     if not (min_len <= n_in_cluster <= max_len):
    #         continue
    #     ids = cn.pre_order(lambda x: x.id)
    #     labels = [all_labels[i] for i in ids]
    #     d = Z[cn.get_id() - ns, 2]
    #     print(f'{cn.get_id()} --> {n_in_cluster} --> {ids} -- > {labels}')
    #     print('distance = ', d)

    # print('-------------------------------------')

    # dists is a dicionary of cluster_id: distance. Distance is fetch from Z

    dists = {cn.get_id(): Z[cn.get_id() - ns, 2] for cn in cn_list[ns:]}

    # Calculate discriminating distances of a set of samples with the same label
    # store in dictionary `discrims`.
    # `total` accumulates total.

    total = 0
    discrims = {label: 0.0 for label in unique_labels}
    # Z[-1,2] is the maximum distance between any 2 clusters
    max_dist = Z[-1,2]

    for cn in cn_list:
        i = cn.get_id()
        len_cluster = cn.get_count()
        # skip if cluster too short or too long
        if not (min_len <= len_cluster <= max_len):
            continue

        labelset = [all_labels[loc] for loc in cn.pre_order(lambda x: x.id)]
        # get first element
        label0 = labelset[0]

        # check if cluster length == exactely the number of samples of label of 1st element.
        # If so, all labels must also be equal
        if len_cluster != sample_number[label0] or labelset.count(label0) != len_cluster:
            continue

        # Compute distances.
        # find iteration when cluster i was integrated in a larger one
        itera = np.where(Z == i)[0][0]
        dif_dist = Z[itera, 2]

        discrims[label0] = (dif_dist - dists[i]) / max_dist
        total += discrims[label0]
        # print(f'\n-----------\ncluster {i}, label set ----> {labelset}')
        # print('discrims ---->', discrims)
        # print('total so far ---->', total)

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
