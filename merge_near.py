import numpy as np
import pandas as pd

# Try to reduce repetition of features with near m/z
# Features within ppmtol and "complementary" get merged

store = pd.HDFStore('alignments.h5')
data_pos = store.get('groups_1ppm_min2_all_1ppm_pos')
data_neg = store.get('groups_1ppm_min2_all_1ppm_neg')

ppmtol = 3

def merge_near_mz(df, ppmtol):
    mz = df.index

    # moving window of consecutive values
    count = 0
    datarows = []
    new_mzs = []
    was_merged = False
    for i, value in enumerate(mz[:-1]):
        if was_merged:
            was_merged = False
            continue
        # is it near?
        m1 = float(value)
        m2 = float(mz[i+1])
        d = 1e6 * (m2-m1) / m1
        if d <= ppmtol:
            # are they "complementary" ?
            row1 = df.iloc[i]
            row2 = df.iloc[i+1]
            intersection = row1.notna() & row2.notna()
            if intersection.sum() == 0:
                count += 1
                # merge rows
                merged = row1.combine_first(row2)
                new_mzs.append(np.mean((m1,m2)))
                datarows.append(merged.values)
                was_merged = True
                # if count < 5:
                #     print(f'---\n{m1:.6f}\n{m2:.6f} : {d:.3f} ppm')
                #     print(row1)
                #     print('-------')
                #     print(row2)
                #     print('------merged-')
                #     print(merged)
        else:
            datarows.append(df.iloc[i].values)
            new_mzs.append(m1)
            if i + 1 == len(df)-1:
                new_mzs.append(m2)
                datarows.append(df.iloc[-1].values)

    print(f'{count} are near')

    newdf = pd.DataFrame(np.array(datarows), index=new_mzs, columns = df.columns)
    newdf.index.names = df.index.names
    newdf.columns.names = df.columns.names
    return newdf

filtered_dataneg = merge_near_mz(data_neg, ppmtol)
filtered_dataneg = merge_near_mz(filtered_dataneg, ppmtol)

filtered_datapos = merge_near_mz(data_pos, ppmtol)
filtered_datapos = merge_near_mz(filtered_datapos, ppmtol)

data_neg.info()
filtered_dataneg.info()

data_pos.info()
filtered_datapos.info()

store.put('groups_1ppm_min2_all_3ppm_pos_clean', filtered_datapos, format='table')
store.put('groups_1ppm_min2_all_3ppm_neg_clean', filtered_dataneg, format='table')