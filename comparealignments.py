# compare "complete linkage" and hc median alignments

import pandas as pd
from collections import OrderedDict
from pathlib import Path

from metabolinks import align, read_data_from_xcel
from metabolinks.similarity import mz_similarity
import metabolinks as mtl
pd.set_option('io.hdf.default_format','table')

print('----alignments in previous store--------')

alignments = pd.HDFStore('alignments.h5')

for k in alignments.keys():
    print(k)
print('------- alignments old algorithm medians ---------------')
alignments_old = pd.HDFStore('alignments_old.h5')

for k in alignments_old.keys():
    print(k)

print('---------alignments new algorithm -------------')
alignments_new = pd.HDFStore('alignments_new.h5')

for k in alignments_new.keys():
    print(k)

print('-------- old store --------------')

oppm1min2ppm1 = alignments.get('groups_1ppm_min2_all_1ppm_neg')
oppm1min2ppm1.info()

print('--------- old algorithm -------------')

nppm1min2ppm1 = alignments_old.get('groups_1ppm_min2_all_1ppm_neg')
nppm1min2ppm1.info()

print('---------- new algorithm ------------')

nppm1min2ppm1 = alignments_new.get('groups_1ppm_min2_all_1ppm_neg')
nppm1min2ppm1.info()


print('----------------------')
for k in list(alignments_new.keys()):
    if k.endswith('clean'):
        alignments_new.remove(k)