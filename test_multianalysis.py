""" Tests for module multianalysis."""
import pytest
from pytest import approx

import scaling as sca
import multianalysis as ma
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

import metabolinks as mtl


# Tests for dist_discrim

def reading_MetAna_file(filename, has_labels=False):
    if has_labels:
        df = pd.read_csv(filename, header=[0,1], sep=',', index_col=0)
        # data has labels, but they are in the inner level of columns. Push to outer and rename levels
        df = df.swaplevel(axis=1).rename_axis(['label', 'sample'], axis='columns')
    else:
        df = pd.read_csv(filename, header=0, sep=',', index_col=0).rename_axis('sample', axis='columns')
    df = df.rename_axis('mz/rt', axis='index')
    # these may exist, repeating information
    df = df.drop(columns=["mz", "rt"], errors='ignore')
    if not has_labels:
        # force labels
        df = mtl.add_labels(df, labels=['KO', 'WT'])
    return df

aligned_all_neg = reading_MetAna_file('aligned_1ppm_min2_1ppm_negative.csv', has_labels=True)

preprocessed = sca.NaN_Imputation(aligned_all_neg, 0).pipe(sca.ParetoScal)

dist_euclidian = dist.pdist(preprocessed.T, metric='euclidean')
Z_euc = hier.linkage(dist_euclidian, method='average')

# global_dist, discrims = ma.dist_discrim(aligned_all_neg, Z_euc, method='average')
# print(global_dist, discrims)

def test_dist_dicrim_average():
    global_dist, discrims = ma.dist_discrim(aligned_all_neg, Z_euc, method='average')
    # assert str(discrim_ave[0]) == str(np.array(list(discrim_ave[1].values())).mean())
    assert global_dist == approx(np.array(list(discrims.values())).mean())


def test_dist_dicrim_median():
    global_dist, discrims = ma.dist_discrim(aligned_all_neg, Z_euc, method='median')
    assert global_dist == approx(np.median(list(discrims.values())))