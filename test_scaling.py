import pytest
from pytest import approx
from pandas.testing import assert_frame_equal

import scaling as sca
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

import metabolinks as mtl

# Test file for the module scaling.
# Needs files from MetaboAnalyst besides the ones in this repository (see MetAnalyst_Example.ipynb)

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

MetAna_O2 = reading_MetAna_file('MetAnalyst/MetAna_Original.csv', has_labels=True)
print('ORIGINAL ----------------------------------')
print(MetAna_O2)
print('IMPUTED ----------------------------------')
MetAna_I = reading_MetAna_file('MetAnalyst/MetAna_Imputed.csv', has_labels=True)
print(MetAna_I)
print('NORMALIZED ----------------------------------')
MetAna_N = reading_MetAna_file('MetAnalyst/MetAna_Norm.csv')
print(MetAna_N)
print('PARETO SCALED ONLY ----------------------------------')
MetAna_P = reading_MetAna_file('MetAnalyst/MetAna_Pareto.csv')  # Pareto Scaling only
print(MetAna_P)

MetAna_I2T = pd.read_csv('MetAnalyst/MetAna_Imputed2.csv', header=0, sep=',', index_col=0)
print('IMPUTED TRANSPOSED ----------------------------------')
print(MetAna_I2T)

new_index = MetAna_O2.index.str.split('/').str[0].astype(float)
#print(new_index)
MetAna_O = pd.DataFrame(MetAna_O2.values, index=new_index, columns=MetAna_O2.columns)
print('ORIGINAL, INDEX HAS ONLY M/Z ----------------------------------')
print(MetAna_O)

print('aligned_1ppm_min2_1ppm_negative.csv ----------------------------------')
aligned_all_neg = reading_MetAna_file('aligned_1ppm_min2_1ppm_negative.csv', has_labels=True)
print(aligned_all_neg)

def test_NaN_FeatRemove():
    assert len(sca.NaN_Imputation(MetAna_O2, 1/2 + 0.00001)) == len(MetAna_I2T.columns)-1


def test_NaN_MinValue():
    Imputated = sca.NaN_Imputation(MetAna_O2, minsample=0)
    minimum = (MetAna_O2.min().min())/2
    where_null = MetAna_O2.isnull()
    mean_imputed = Imputated[where_null].mean().mean()
    assert mean_imputed == approx(minimum)

def test_Norm_Values():
    """Normalization by a reference feature only - 301/2791.68 (random choice)."""
    Imputated = sca.NaN_Imputation(MetAna_O2, 0)
    norm = sca.Norm_Feat(Imputated, "301/2791.68")
    # assert str(MetAna_N) == str(norm*1000)
    assert_frame_equal(MetAna_N, norm*1000)

# Tests for glog


# Tests for ParetoScal

def test_ParetoScal_values():
    Imputated = sca.NaN_Imputation(MetAna_O2, 0)
    pareto = sca.ParetoScal(Imputated)
    assert_frame_equal(MetAna_P, pareto)


# Tests for search_for_ref_feat

def test_Ref_Feat_finding2():
    RefEst_Neg = sca.search_for_ref_feat(aligned_all_neg, 554.2615)
    assert RefEst_Neg[0] == approx(554.26202000000001)


def test_Ref_Feat_finding():
    RefEst_Neg = sca.search_for_ref_feat(MetAna_O, 300.5)
    assert RefEst_Neg[0] == approx(301)

# Tests for dist_discrim
# Imputated_neg = sca.NaN_Imputation(aligned_all_neg, 0)
# Euc_neg = sca.ParetoScal(Imputated_neg)
preprocessed = sca.NaN_Imputation(aligned_all_neg, 0).pipe(sca.ParetoScal)

dist_euclidian = dist.pdist(preprocessed.T, metric='euclidean')
Z_euc = hier.linkage(dist_euclidian, method='average')

#discrim = sca.dist_discrim(aligned_all_neg, Z_euc, method='average')

def test_dist_dicrim_average():
    global_dist, discrims = sca.dist_discrim(aligned_all_neg, Z_euc, method='average')
    # assert str(discrim_ave[0]) == str(np.array(list(discrim_ave[1].values())).mean())
    assert global_dist == approx(np.array(list(discrims.values())).mean())


def test_dist_dicrim_median():
    global_dist, discrims = sca.dist_discrim(aligned_all_neg, Z_euc, method='median')
    assert global_dist == approx(np.median(list(discrims.values())))
