# Alignment of peak lists based on m/z relative differences (below a ppm threshold). 
# 
# Requirements:
# 
# - metabolinks
# 
import sys

import pandas as pd
from collections import OrderedDict
from pathlib import Path

from metabolinks import align, read_data_from_xcel
from metabolinks.similarity import mz_similarity
import metabolinks as mtl

if len(sys.argv) < 2 or sys.argv[1] == 'hc':
    alignments = pd.HDFStore('alignments_new.h5',complevel=9, complib='blosc:lz4')
    grouping = 'hc'
elif sys.argv[1] == 'complete':
    alignments = pd.HDFStore('alignments_old.h5',complevel=9, complib='blosc:lz4')
    grouping = 'complete'
else:
    raise ValueError(f'Invalid argument {sys.argv[1]}')


pd.set_option('io.hdf.default_format','table')

print(alignments.keys())

# ### Set up metadata descriptions

data_folder = 'data'
header_row = 3

data = {
    'CAN': {'filename': 'CAN (14, 15, 16).xlsx',
            'names'   : {'sample_names': '14 15 16'.split(), 'labels' : 'CAN'}},
    'CS':  {'filename': 'CS (29, 30, 31).xlsx',
            'names'   : {'sample_names': '29 30 31'.split(), 'labels' : 'CS'}},
    'LAB':  {'filename': 'LAB (8, 9, 10).xlsx',
            'names'   : {'sample_names': '8  9  10'.split(), 'labels' : 'LAB'}},
    'PN':  {'filename': 'PN (23, 24, 25).xlsx',
            'names'   : {'sample_names': '23 24 25'.split(), 'labels' : 'PN'}},
    'REG':  {'filename': 'REG (38, 39, 40).xlsx',
            'names'   : {'sample_names': '38 39 40'.split(), 'labels' : 'REG'}},
    'RIP':  {'filename': 'RIP (17, 18, 19).xlsx',
            'names'   : {'sample_names': '17 18 19'.split(), 'labels' : 'RIP'}},
    'RL':  {'filename': 'RL (26, 27, 28).xlsx',
            'names'   : {'sample_names': '26 27 28'.split(), 'labels' : 'RL'}},
    'ROT':  {'filename': 'ROT (20, 21, 22).xlsx',
            'names'   : {'sample_names': '20 21 22'.split(), 'labels' : 'ROT'}},
    'RU':  {'filename': 'RU (35, 36, 37).xlsx',
            'names'   : {'sample_names': '35 36 37'.split(), 'labels' : 'RU'}},
    'SYL':  {'filename': 'SYL (11, 12, 13).xlsx',
            'names'   : {'sample_names': '11 12 13'.split(), 'labels' : 'SYL'}},
    'TRI':  {'filename': 'TRI (32, 33, 34).xlsx',
            'names'   : {'sample_names': '32 33 34'.split(), 'labels' : 'TRI'}},
    # these are the new cultivars
    # 'CFN':  {'filename': 'CFN (10713_1, 10713_2, 10713_3).xlsx',
    #         'names'   : {'sample_names': '10713-1 10713-2 10713-3'.split(), 'labels' : 'CFN'}},
    # 'CHT':  {'filename': 'CHT (13514_1, 13514_2, 13514_3).xlsx',
    #         'names'   : {'sample_names': '13514-1 13514-2 13514-3'.split(), 'labels' : 'CHT'}},
    # 'SB':  {'filename': 'SB (53211_1, 53211_2, 53211_3).xlsx',
    #         'names'   : {'sample_names': '53211-1 53211-2 53211-3'.split(), 'labels' : 'SB'}},
}

# ### Read spectra from Excel files

def read_vitis_data(filename, metadata):
    exp=read_data_from_xcel(filename, header=[3])
    for sname in exp:
        dfs = exp[sname]
        label2assign = metadata['names']['labels']
        for name, df in zip(metadata['names']['sample_names'], dfs):
            df.columns = [name]
            df.index.name = 'm/z'
        exp[sname] = [mtl.add_labels(df, labels=label2assign) for df in exp[sname]]
    return exp
# exp = read_vitis_data(f"data/{data['CAN']['filename']}", data['CAN'])
#exp # seems ok!

print('============= Reading all spectra ============')
all_spectra = OrderedDict()

for d, desc in data.items():
    fpath = Path(data_folder, desc['filename'])
    sheets = read_vitis_data(fpath, desc)
    for sheet, spectra in sheets.items():
        print(f'Sheet {sheet} contains {len(spectra)} spectra')
        all_spectra[sheet] = spectra

# ### Alignment of peak lists
# #### Align for each mode and cultivar (keep if peak appears in at least 2 samples)

def align_each_sheet_then_globally(all_spectra, ppmtol, min_samples, ppmtol_global, min_samples_global):

    aligned = {}
    for k, s in all_spectra.items():
        print('=======================================')
        print(k)
        # print(s)
        aligned[k]  = align(s, ppmtol, min_samples, grouping=grouping)
    
    # Separate modes
    aligned_pos = {name : value for name,value in aligned.items() if name.upper()[-8:-1]=='POSITIV'}
    aligned_neg = {name : value for name,value in aligned.items() if name.upper()[-8:-1]=='NEGATIV'}

    #save_aligned_to_excel('aligned_cultivars_positive_1ppm_min2.xlsx', aligned_pos)
    #save_aligned_to_excel('aligned_cultivars_negative_1ppm_min2.xlsx', aligned_neg)

    # Align globally the previously obtained alignments (for each mode).

    positive = aligned_pos.values()
    negative = aligned_neg.values()

    aligned_all_pos = align(positive, ppmtol=ppmtol_global, min_samples=min_samples_global,
                            grouping=grouping)
    aligned_all_neg = align(negative, ppmtol=ppmtol_global, min_samples=min_samples_global,
                            grouping=grouping)
    return (aligned_all_pos, aligned_all_neg)

aligned_all_pos, aligned_all_neg = align_each_sheet_then_globally(all_spectra,
                                                                  1.0,
                                                                  2,
                                                                  1.0,
                                                                  1)

# ### Test hdf5 store (writting and reading back, using `put` and `get`)
# Other functions are `df.to_hdf(store)` and `store.append(key, df)`
# alignments.put('groups_1ppm_min2_all_1ppm_pos', aligned_all_pos)
# it seems to work
# bigalignment = alignments.get('groups_1ppm_min2_all_1ppm_neg')
# bigalignment.info()

# Nomenclature: first groups at 1ppm then all at 1ppm
alignments.put('groups_1ppm_min2_all_1ppm_neg', aligned_all_neg)
alignments.put('groups_1ppm_min2_all_1ppm_pos', aligned_all_pos)


aligned_all_pos, aligned_all_neg = align_each_sheet_then_globally(all_spectra,
                                                                  1.0,
                                                                  3,
                                                                  1.0,
                                                                  1)

# Nomenclature: first groups at 1ppm then all at 1ppm
alignments.put('groups_1ppm_min3_all_1ppm_neg', aligned_all_neg)
alignments.put('groups_1ppm_min3_all_1ppm_pos', aligned_all_pos)

aligned_all_pos, aligned_all_neg = align_each_sheet_then_globally(all_spectra,
                                                                  2.0,
                                                                  2,
                                                                  2.0,
                                                                  1)

# Nomenclature: first groups at 1ppm then all at 1ppm
alignments.put('groups_2ppm_min2_all_2ppm_neg', aligned_all_neg)
alignments.put('groups_2ppm_min2_all_2ppm_pos', aligned_all_pos)

aligned_all_pos, aligned_all_neg = align_each_sheet_then_globally(all_spectra,
                                                                  2.0,
                                                                  3,
                                                                  2.0,
                                                                  1)

# Nomenclature: first groups at 1ppm then all at 1ppm
alignments.put('groups_2ppm_min3_all_2ppm_neg', aligned_all_neg)
alignments.put('groups_2ppm_min3_all_2ppm_pos', aligned_all_pos)


ppmtol = 1.0 #2.0 for the groups_2ppm_min3_all_2ppm_neg/pos; 1 for the rest.
min_samples = 2 #2 for the groups_1ppm_min2_all_1ppm_neg/pos, 3 for the groups_1ppm_min3_all_1ppm_neg/pos and 
                #groups_2ppm_min3_all_2ppm_neg/pos

ppmtol_global = 1.0 #2.0 for the groups_2ppm_min3_all_2ppm_neg/pos
min_samples_global = 1 #Now it has to be 1

# ### Aligning all 39 samples together (not aligning replicates first)
 
# First, putting all samples in the same list with the correct sample_names

# Separate modes
pos = []
neg = []
for k, s in all_spectra.items():
    if k.upper()[-8:-1]=='POSITIV':
        pos.extend(s)
    elif k.upper()[-8:-1]=='NEGATIV':
        neg.extend(s)
    else:
        pass


ppmtol = 1.0
min_samples = 2 # 2 for the all_1ppm_min2_neg/pos, 6 for the all_1ppm_min6_neg/pos and 13 for the all_1ppm_min13_neg/pos

aligned_all_positive = align(pos, 1.0, 2, grouping=grouping)
aligned_all_negative = align(neg, 1.0, 2, grouping=grouping)
# Nomenclature: all samples at 1ppm with n min_samples
alignments.put('all_1ppm_min2_pos', aligned_all_positive)
alignments.put('all_1ppm_min2_neg', aligned_all_negative)

alignments.close()