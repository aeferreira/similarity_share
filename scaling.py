import pandas as pd
from metabolinks import AlignedSpectra

"""Elimination of features with too many missing values, missing value estimation and Pareto Scaling of data"""


def NaN_Imputation(Spectra, minsample=0):
    """Remove features with too many missing values (0 < minsample < 1 = % of samples where the feature must be present)
       Replace missing values by half of the minimum intensity of the original data in that sample

       Requires: minsample between 0 and 1"""

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
    Imputated.data.fillna(Imputated.data.min()/2, inplace=True)
    return Imputated


# Function to do Pareto Scaling, it accomodates Missing Values.
# It is a bit inneficient (last 'for' can be incorporated in the one before if it doesn't accomodate Missing Values)
def ParetoScal(Spectra):
    """Performs Pareto Scaling on an AlignedSpectra object"""

    scaled_aligned = Spectra.data.copy()
    for j in range(0, Spectra.sample_count):
        std = Spectra.sample(j).data.std()[0]
        sqstd = std**(0.5)
        values = Spectra.sample(j).data
        scaled = []
        # Apply Pareto Scaling to each value
        for i in range(0, len(Spectra.sample(j))):
            scaled.append((values.iloc[i, 0] - values.mean()[0])/sqstd)
        # Replace not null values by the scaled values
        if len(scaled) == len(scaled_aligned):
            scaled_aligned.iloc[:, j] = scaled
        else:
            a = 0
            for i in range(0, len(scaled_aligned)):
                if scaled_aligned.notnull().iloc[i, ].at[values.columns[0]]:
                    scaled_aligned.iloc[i, ].at[values.columns[0]] = scaled[a]
                    a = a + 1

    # Return scaled spectra
    return AlignedSpectra(scaled_aligned, sample_names=Spectra.sample_names, labels=Spectra.labels)
