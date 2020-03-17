""" Tests for module scaling."""
import pytest
from pytest import approx
from pandas.testing import assert_frame_equal, assert_series_equal

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

import metabolinks as mtl
import scaling as sca

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


# A slightly modified version of "original" dataset
# The row index was parsed to retain m/z values only, as floats
new_index = MetAna_O2.index.str.split('/').str[0].astype(float)

MetAna_O = pd.DataFrame(MetAna_O2.values, index=new_index, columns=MetAna_O2.columns)
print('ORIGINAL, INDEX HAS NOW ONLY M/Z ----------------------------------')
print(MetAna_O)

print('aligned_1ppm_min2_1ppm_negative.csv ----------------------------------')
aligned_all_neg = reading_MetAna_file('aligned_1ppm_min2_1ppm_negative.csv', has_labels=True)
print(aligned_all_neg)

def test_NaN_FeatRemove():
    assert len(sca.NaN_Imputation(MetAna_O2, 1/2 + 0.00001)) == len(MetAna_I2T.columns)-1


def test_NaN_MinValue():
    imputated = sca.NaN_Imputation(MetAna_O2, minsample=0)
    minimum = (MetAna_O2.min().min())/2
    where_null = MetAna_O2.isnull()
    mean_imputed = imputated[where_null].mean().mean()
    assert mean_imputed == approx(minimum)


def test_Norm_Values():
    """Normalization by a reference feature only - 301/2791.68 (random choice)."""
    Imputated = sca.NaN_Imputation(MetAna_O2, 0)
    norm = sca.Norm_Feat(Imputated, "301/2791.68")
    # assert str(MetAna_N) == str(norm*1000)
    assert_frame_equal(MetAna_N, norm*1000)

# Tests for Norm_TotalInt


# Tests for Norm_PQN


# Tests for Norm_Quantile


# Tests for glog
MetAna_G = reading_MetAna_file('MetAnalyst/MetAna_Glog.csv')  # Pareto Scaling only


def test_glog_values():
    glog = sca.glog(MetAna_O2_I)
    # assert str(MetAna_G.data) == str(glog.data)
    assert_frame_equal(MetAna_G, glog)


def test_glog_lamb():
    lamb = 100000
    y = MetAna_O2_I.copy()
    y = np.log2((y + (y**2 + lamb**2)**0.5)/2)
    assert (y == (sca.glog(MetAna_O2_I, lamb = 100000))).all().all()


# Tests for ParetoScal

MetAna_O2_I = sca.NaN_Imputation(MetAna_O2, 0)

def test_ParetoScal_values():
    Imputated = sca.NaN_Imputation(MetAna_O2, 0)
    pareto = sca.ParetoScal(Imputated)
    assert_frame_equal(MetAna_P, pareto)


# Tests for MeanCentering
# If Pareto and AutoScale work, MeanCentering should work

# Tests for AutoScal
MetAna_AS = reading_MetAna_file('MetAnalyst/MetAna_AutoScal.csv')  # Pareto Scaling only


def test_AutoScal_values():
    auto = sca.AutoScal(MetAna_O2_I)
    # assert str(MetAna_AS) == str(auto)
    assert_frame_equal(MetAna_AS, auto)

# Tests for RangeScal

MetAna_RS = reading_MetAna_file('MetAnalyst/MetAna_RangeScal.csv')  # Pareto Scaling only

# Add row with same maximum and minimum value as well as missing values - the row should stay the same according to MetaboAnalyst
# Maybe it should just be zero and missing values but that's a topic for another day
# TODO: OK, for another day it is!

MetAna_O2_I_RS = MetAna_O2_I.copy()
MetAna_O2_I_RS.loc['205/2790.89'] = [100000, 100000, np.nan, np.nan, 100000, 100000, np.nan, np.nan, 100000, 100000, np.nan, np.nan]

def test_RangeScal_values():
    ranges = sca.RangeScal(MetAna_O2_I)
    # assert str(MetAna_RS) == str(ranges)
    assert_frame_equal(MetAna_RS, ranges)

def test_RangeScal_ranges():
    "Testing invariance for features with null range."
    ranges = sca.RangeScal(MetAna_O2_I_RS)
    # assert str(ranges.loc['205/2790.89']) == str(MetAna_O2_I_RS.loc['205/2790.89'])
    assert_series_equal(ranges.loc['205/2790.89'], MetAna_O2_I_RS.loc['205/2790.89'])

# Tests for VastScal


# Tests for LevelScal


# Tests for search_for_ref_feat

def test_Ref_Feat_finding2():
    RefEst_Neg = sca.search_for_ref_feat(aligned_all_neg, 554.2615)
    assert RefEst_Neg[0] == approx(554.26202000000001)


def test_Ref_Feat_finding():
    RefEst_Neg = sca.search_for_ref_feat(MetAna_O, 300.5)
    assert RefEst_Neg[0] == approx(301)


