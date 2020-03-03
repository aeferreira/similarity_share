import pytest
import scaling as sca
from metabolinks import read_aligned_spectra, AlignedSpectra
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
import numpy as np

# Test file for the module scaling.

MetAna_O2 = read_aligned_spectra('MetAnalyst/MetAna_Original.csv', labels=True, sep=',')
a = []
O = MetAna_O2.data.copy()
for i in range(len(O.index)):
    a.append(O.index[i].split('/'))
    O.rename(index={O.index[i]: float(a[i][0])}, inplace=True)
MetAna_O = AlignedSpectra(O, labels=MetAna_O2.labels)

MetAna_I = read_aligned_spectra('MetAnalyst/MetAna_Imputed.csv', labels=True, sep=',')


def reading_MetAna_files(filename):
    file = pd.read_table(filename, header=[0], sep=',')
    file = file.set_index(file.columns[0])
    file.index.name = 'm/z'
    file = file[["ko15", "ko16", "ko18", "ko19", "ko21",
                 "ko22", "wt15", "wt16", "wt18", "wt19", "wt21", "wt22"]]
    MetAna_file = AlignedSpectra(file, labels=[
                                 "KO", "KO", "KO", "KO", "KO", "KO", "WT", "WT", "WT", "WT", "WT", "WT"])
    return MetAna_file


# Tests for NaN_Imputation
MetAna_I2T = read_aligned_spectra(
    'MetAnalyst/MetAna_Imputed2.csv', labels=True, sep=',')  # Transposed


def test_NaN_FeatRemove():
    assert len(sca.NaN_Imputation(MetAna_O2, 1/2 +
                                  0.00001).data) == len(MetAna_I2T.data.columns)-1


def test_NaN_MinValue():
    Imputated = sca.NaN_Imputation(MetAna_O2, 0)
    minimum = min(MetAna_O2.data.min()/2)
    Bool = MetAna_O2.data.isnull()
    values = Imputated.data[Bool == True].mean().mean()
    assert str(values)[0:9] == str(minimum)[0:9]


# Tests for remove_feat
def test_FeatRemove():
    assert len(sca.remove_feat(MetAna_O2, 1/2 +
                                  0.00001).data) == len(MetAna_I2T.data.columns)-1


# Base file for most normalizations, transformations and scaling methods
MetAna_O2_I = sca.NaN_Imputation(MetAna_O2, 0) #Original file from MetaboAnalyst after NaN_Imputation function


# Tests for Norm_Feat
# Normalization by a reference feature only - 301/2791.68 (random choice)
MetAna_N = reading_MetAna_files('MetAnalyst/MetAna_Norm.csv')


def test_Norm_Values():
    norm = sca.Norm_Feat(MetAna_O2_I, "301/2791.68")
    assert str(MetAna_N.data) == str(norm.data*1000)


# Tests for Norm_TotalInt


# Tests for Norm_PQN


# Tests for Norm_Quantile


# Tests for glog
MetAna_G = reading_MetAna_files('MetAnalyst/MetAna_Glog.csv')  # Pareto Scaling only


def test_glog_values():
    glog = sca.glog(MetAna_O2_I)
    assert str(MetAna_G.data) == str(glog.data)


def test_glog_lamb():
    lamb = 100000
    y = MetAna_O2_I.data.copy()
    y = np.log2((y + (y**2 + lamb**2)**0.5)/2)
    assert (y == (sca.glog(MetAna_O2_I, lamb = 100000).data)).all().all()


# Tests for ParetoScal
MetAna_P = reading_MetAna_files('MetAnalyst/MetAna_Pareto.csv')  # Pareto Scaling only


def test_ParetoScal_values():
    pareto = sca.ParetoScal(MetAna_O2_I)
    assert str(MetAna_P) == str(pareto)


# Tests for MeanCentering


# Tests for AutoScal
MetAna_AS = reading_MetAna_files('MetAnalyst/MetAna_AutoScal.csv')  # Pareto Scaling only


def test_AutoScal_values():
    auto = sca.AutoScal(MetAna_O2_I)
    assert str(MetAna_AS) == str(auto)

# Tests for RangeScal
MetAna_RS = reading_MetAna_files('MetAnalyst/MetAna_RangeScal.csv')  # Pareto Scaling only
MetAna_O2_I_RS = MetAna_O2_I.data.copy()
# Add row with same maximum and minimum value as well as missing values - the row should stay the same according to MetaboAnalyst
# Maybe it should just be zero and missing values but that's a topic for another day
MetAna_O2_I_RS.loc['205/2790.89'] = [100000, 100000, np.nan, np.nan, 100000, 100000, np.nan, np.nan, 100000, 100000, np.nan, np.nan]


def test_RangeScal_values():
    ranges = sca.RangeScal(MetAna_O2_I)
    assert str(MetAna_RS) == str(ranges)


def test_RangeScal_ranges():
    ranges = sca.RangeScal(AlignedSpectra(MetAna_O2_I_RS))
    assert str(ranges.data.loc['205/2790.89']) == str(MetAna_O2_I_RS.loc['205/2790.89'])


# Tests for VastScal


# Tests for LevelScal


# Tests for search_for_ref_feat
aligned_all_neg = read_aligned_spectra(
    'aligned_1ppm_min2_1ppm_negative.csv', labels=True, sep=',')


def test_Ref_Feat_finding():
    RefEst_Neg = sca.search_for_ref_feat(aligned_all_neg, 554.2615)
    assert RefEst_Neg[0] == 554.26202000000001


def test_Ref_Feat_finding2():
    RefEst_Neg = sca.search_for_ref_feat(MetAna_O, 300.5)
    assert RefEst_Neg[0] == 301


# Tests for dist_discrim
Imputated_neg = sca.NaN_Imputation(aligned_all_neg, 0)
Euc_neg = sca.ParetoScal(Imputated_neg)
dist_euc_neg = dist.pdist(Euc_neg.data.T, metric='euclidean')
Z_euc_neg = hier.linkage(dist_euc_neg, method='average')


def test_dist_dicrim_average():
    discrim = sca.dist_discrim(aligned_all_neg, Z_euc_neg, method='average')
    assert str(discrim[0]) == str(np.array(list(discrim[1].values())).mean())


def test_dist_dicrim_median():
    discrim = sca.dist_discrim(aligned_all_neg, Z_euc_neg, method='median')
    assert discrim[0] == np.median(list(discrim[1].values()))
