# Needed imports
import pandas as pd
import metabolinks as mtl
import metabolinks.transformations as trans
import numpy as np

"""Here are compiled the functions developed for traditional data pre-treatments in metabolomics data analysis workflow. The functions
are split in the following sub-sections:
- Feature Filtering: filtering features based on min. nº of samples they appear and imputation of missing values (NaN_Imputation).
- Normalizations: Normalization by a reference feature (Norm_Feat), by the total sum of peaks (Norm_TotalInt), Probabilistic Quotient
Normalization (Norm_PQN) and Quantile Normalization (Norm_Quantile).
- Transformations: Generalized Logarithmic Transformation (glog).
- Centering and Scaling: Pareto Scaling (ParetoScal), Mean Centering (MeanCentering), Auto Scaling (AutoScal), Range Scaling
(RangeScal), Vast Scaling (VastScal) and Level Scaling (LevelScal)
- Miscellaneous functions

Some of these functions were then included in the transformations module of the metabolinks Python package and are applied here (now)
by retrieving said functions from that module, namely NaN_Imputation (split into 2), Norm_Feat, glog and ParetoScal.
"""

### ---------- Missing Value Imputation and Feature Filtering ----------
def NaN_Imputation(df, minsample=0):
    "Feature filtering (based on min. nº of samples they appear) and imputes remaining missing values (fraction of min. value in data)."
    df = trans.keep_atleast(df, min_samples=minsample)
    df = trans.fillna_frac_min(df, fraction=0.5)
    return df


### ---------- Normalizations --------------
def Norm_Feat(df, Feat_mass, remove=True):
    "Normalization of a dataset by a reference feature."
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
    # "Building" the reference sample based on the input given
    if ref_sample == 'mean': # Mean spectra of all samples
        ref_sample2 = df.T / df.mean(axis = 1)
    elif ref_sample == 'median': # Median spectra of all samples
        ref_sample2 = df.T / df.median(axis = 1)
    elif ref_sample in df.columns: # Specified sample of the spectra. ('Label','Sample') if data is labelled (multi index)
        ref_sample2 = df.T / df.loc[:,ref_sample]
    else: # Actual sample given
        ref_sample2 = df.T / ref_sample

    # Normalization Factor and Normalization
    Norm_fact = ref_sample2.median(axis=1)

    return df / Norm_fact


def Norm_Quantile(df, ref_type='mean'):
    """Normalization of a dataset by the Quantile Normalization method.

       Missing Values are temporarily replaced with 0 (and count as 0) until normalization is done. Quantile Normalization is more
    useful with no/low number of missing values.

       df: Pandas DataFrame.
       ref_type: str (default: 'mean'); reference sample to use in Quantile Normalization, types accepted: 'mean' (default,
    reference sample will be the means of the intensities of each rank), 'median' (reference sample will be the medians of the
    intensities for each rank).

       Returns: Pandas DataFrame; normalized spectra.
    """
    # Setting up the temporary dataset with missing values replaced by zero and dataframes for the results
    norm = df.copy().replace({np.nan:0})
    ref_spectra = df.copy()
    ranks = df.copy()

    for i in range(len(norm.columns)):
        # Determining the ranks of each entry in the same row (same variable) in the dataset
        ref_spectra.iloc[:,i] = norm.iloc[:,i].sort_values().values
        ranks.iloc[:,i] = norm.iloc[:,i].rank(na_option='top')

    # Determining the reference sample for normalization based on the ref_type chosen (applied on the columns).
    if ref_type == 'mean':
        ref_sample = ref_spectra.mean(axis=1).values
    elif ref_type == 'median':
        ref_sample = ref_spectra.median(axis=1).values
    else:
        raise ValueError('Type not recognized. Available ref_type: "mean", "median".')

    # Replacing the values in the dataset for the reference sample values based on the ranks calculated earlier for each entry
    for i in range(len(ranks)):
        for j in range(len(ranks.columns)):
            if ranks.iloc[i,j] == round(ranks.iloc[i,j]):
                norm.iloc[i,j] = ref_sample[int(ranks.iloc[i,j])-1]
            else: # in case the rank isn't an integer and ends in .5 (happens when a pair number of samples have the same
                  # value in the same column - after ordering from lowest to highest values by row).
                norm.iloc[i,j] = np.mean((ref_sample[int(ranks.iloc[i,j]-1.5)], ref_sample[int(ranks.iloc[i,j]-0.5)]))

    # Replacing 0s by missing values and creating the DataFrame for the output
    return norm.replace({0:np.nan})


### ---------- Transformations --------------
def glog(df, lamb=None):
    """Generalized Logarithmic Transformation on a dataset (same as MetaboAnalyst's transformation).

    df: Pandas DataFrame.
    lamb: scalar, optional (default: global minimum divided by 10); transformation parameter lambda.

    Returns: Pandas DataFrame; Dataset transformed as log2(y + (y**2 + lamb**2)**0.5)."""

    return trans.glog(df, lamb=lamb)


### ---------- Centering and Scalings --------------
# (Accomodates Missing Values)
def ParetoScal(df):
    "Pareto Scaling of a dataset."
    return trans.pareto_scale(df)

def MeanCentering(df):
    """Mean Centering of a dataset.

       df: Pandas DataFrame. It can include missing values.

       Returns: Pandas DataFrame; Mean Centered Spectra."""
    return df.sub(df.mean(axis=1), axis=0)


def AutoScal(df):
    """Autoscaling of a dataset.

       This is x -> (x - mean(x)) / std(x) per feature.

       df: Pandas DataFrame. Can include missing values.

       Returns: Pandas DataFrame; Auto Scaled Spectra."""

    # TODO: verify if the name of this transformation is "Standard scaling"
    # TODO: most likely it is known by many names (scikit-learn has a StandardScaler transformer)
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    df2 = df.sub(means, axis=0).div(stds, axis=0)
    return df2

def RangeScal(df):
    """Range Scaling of a dataset.

       df: Pandas DataFrame. It can include missing values.

       Returns: Pandas DataFrame; Ranged Scaled Spectra."""

    scaled_aligned = df.copy()
    ranges = df.max(axis=1) - df.min(axis=1) # Defining range for every feature

    # Applying Range scaling to each feature
    for j in range(0, len(scaled_aligned)):
        if ranges.iloc[j] == 0: # No difference between max and min values
            scaled_aligned.iloc[j, :] = df.iloc[j, ]
        else:
            scaled_aligned.iloc[j, :] = (df.iloc[j, ] - df.iloc[j, ].mean()) / ranges.iloc[j]

    # Return scaled spectra
    return scaled_aligned


def VastScal(df):
    """Vast Scaling of a dataset.

       df: Pandas DataFrame. It can include missing values.

       Returns: Pandas DataFrame; Vast Scaled Spectra."""

    std = df.std(axis=1)
    mean = df.mean(axis=1)

    # Applying Vast Scaling to each feature
    scaled_aligned = (((df.T - mean)/std)/(mean/std)).T

    # Return scaled spectra
    return scaled_aligned


def LevelScal(df, average=True):
    """Level Scaling of a dataset.

    Level Scaling made as presented in van den Berg et al., 2006.

    df: Pandas DataFrame. It can include missing values.
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


### ---------- Miscellaneous --------------
def search_for_ref_feat(df, approx_mass):
    """Estimates a peak m/z to be the reference feature in a mass spectrometry dataset.

       df: Pandas DataFrame.
       approx_mass: scalar, approximate mass of the reference feature to search for.

       Return: (ref_feat, dist)
        ref_feat: scalar; peak m/z that is estimated to belong to the reference feature (must be present in all samples, must be at a max
    distance of 1 m/z of the approximate mass given) - the scalar returned is the closest peak that fulfills these two conditios.
        dist: scalar, m/z difference of approximate m/z and estimated m/z.
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

# Outdated
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

       Returns: Processed Pandas DataFrame.
    """
    if minsample != 100: # Missing Value Imputation
        df = NaN_Imputation(df, minsample)

    if Feat_mass is not None: # Normalization by a reference feature
        df = Norm_Feat(df, Feat_mass, remove=remove)

    if do_glog: # glog transformation
        df = glog(df, lamb)

    if Pareto: # Pareto Scaling
        df = ParetoScal(df)

    # Return treated dataset
    return df

