import pandas as pd
from metabolinks import AlignedSpectra
import numpy as np

"""Elimination of features with too many missing values, missing value estimation, normalization by reference feature,
generalized logarithmic transformation, reference feature estimation and Pareto Scaling of data."""


def NaN_Imputation(Spectra, minsample=0):
    """Remove features with a certain % of missing values, replaces remaining ones by half of the minimum value of the original data.

       Spectra: AlignedSpectra object (from metabolinks).
       minsample: scalar, optional; number between 0 and 1, minsample*100 represents the minimum % of samples where the feature must 
    be present in order to not be removed.

       Returns: AlignedSpectra object (from metabolinks); Equal to Spectra but with some features removed and missing values replaced.
    """

    Imputated = Spectra
    df = Imputated.data
    if minsample != 0:
        NumValues = Imputated.data.notnull()
        a = 0
        for i in range(0, len(NumValues)):
            if sum(NumValues.iloc[i, :]) < minsample*Imputated.sample_count:
                # Taking away features that appear in less of minsample% of samples.
                df = df.drop([df.iloc[a].name])
            else:
                a = a + 1

    Imputated = AlignedSpectra(
        df, sample_names=Imputated.sample_names, labels=Imputated.labels)
    # Replace missing values
    Imputated.data.fillna(min(Imputated.data.min()/2), inplace=True)

    return Imputated


def Norm_Feat(Spectra, Feat_mass, remove=True):
    """Normalization by a reference feature.

       Spectra: AlignedSpectra object (from metabolinks).
       Feat_mass: scalar; m/z of the reference feature to normalize the sample.
       remove: bool; True to remove reference feature from data after normalization.

       Returns: AlignedSpectra object (from metabolinks); normalized spectra.
       """

    temp = Spectra.data.copy()
    for i in range(Spectra.sample_count):
        temp.iloc[:, i] = temp.iloc[:, i]/Spectra.data.loc[Feat_mass][i]
    if remove:
        temp = temp.drop([Feat_mass])

    return AlignedSpectra(temp, sample_names=Spectra.sample_names, labels=Spectra.labels)


# Lots of work to do here; optimize lambda and such
# Currently working on it
def glog(Spectra, lamb=0):
    """Performs Generalized Logarithmic Transformation on a Spectra.

       Spectra: AlignedSpectra object (from metabolinks).
       lamb: scalar, optional;  transformation parameter.

       Returns: AlignedSpectra object (from metabolinks); transformed spectra by a factor of log(y + (y**2 + lamb)**0.5).
       """

    y = Spectra.data.copy()
    y = np.log(y + (y**2 + lamb)**0.5)
    return AlignedSpectra(y, sample_names=Spectra.sample_names, labels=Spectra.labels)


# Function to do Pareto Scaling, it accomodates Missing Values.
def ParetoScal(Spectra):
    """Performs Pareto Scaling on an AlignedSpectra object.

       Spectra: Aligned Spectra object (from metabolinks). It can include missing values.

       Returns: Aligned Spectra object (from metabolinks); Pareto Scaled Spectra."""

    scaled_aligned = Spectra.data.copy()
    for j in range(0, len(scaled_aligned)):
        std = Spectra.data.iloc[j, ].std()
        sqstd = std**(0.5)
        values = Spectra.data.iloc[j, ]
        # Apply Pareto Scaling to each value
        values = (values - values.mean())/sqstd
        # Replace not null values by the scaled values
        if len(values) == Spectra.sample_count:
            scaled_aligned.iloc[j, :] = values
        else:
            a = 0
            for i in range(0, len(Spectra.sample_count)):
                if scaled_aligned.notnull().iloc[j, i]:
                    scaled_aligned.iloc[j, i] = values.iloc[a, 0]
                    a = a + 1

    # Return scaled spectra
    return AlignedSpectra(scaled_aligned, sample_names=Spectra.sample_names, labels=Spectra.labels)


def search_for_ref_feat(Spectra, approx_mass):
    """Estimates a peak m/z to be the reference feature. 

       Spectra = AlignedSpectra object (from metabolinks).
       approx_mass = scalar, approximate mass of the reference feature to search for

       Return: scalar, scalar; peak m/z that is estiamted to belong to the reference feature (must be present in all samples,
    must be at a max length of 1 m/z of the approximate mass given) - the scalar returned is the closest peak that fulfills 
    these two conditios; m/z difference of approximate m/z and estimated m/z.
    """
    # Condition 1: Be at a max length of 1 of the approximate mass given.
    rest1 = Spectra.data.copy().index[Spectra.data.index < approx_mass+1]
    rest2 = rest1[rest1 > approx_mass-1]

    # Condition 2: Be present in every sample.
    feat_est = []
    for i in range(len(rest2)):
        if sum(np.isnan(Spectra.data.loc[rest2[i]])) == 0:
            feat_est.append(rest2[i])

    if len(feat_est) == 1:
        return feat_est[0], abs(feat_est[0] - approx_mass)

    elif len(feat_est) == 0:
        return print('Error - No feature is present in all sample around approx_mass')

    # Tiebraker: Be closer to the approximate mass (m/z) given than other peaks that fulfill previous conditions.
    else:
        mass_diff = []
        for i in range(len(feat_est)):
            mass_diff.append(abs(feat_est[i]-approx_mass))

        return feat_est[mass_diff.index(min(mass_diff))], min(mass_diff)
