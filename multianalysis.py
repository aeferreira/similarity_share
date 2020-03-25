import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
import scipy.stats as stats
import sklearn.cluster as skclust
import sklearn.ensemble as skensemble
import random as rd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from matplotlib import cm

import metabolinks as mtl
import metabolinks.transformations as trans

# Apart from mergerank, other functions are incredibly specific to the objectives of this set of notebooks.
# Functions present are for the different kinds of multivariate analysis made in the Jupyter Notebooks

"""Rank creation for hierarchical clustering linkage matrices, discrimination distance for k-means clustering and AHC,
oversampling based on a simple SMOTE method, different application of Random Forests to Spectra data, calculation
of correlation coefficient between linkage matrices of two hierarchical clusterings."""

def dist_discrim(df, Z, method='average'):
    """Give a measure of the normalized distance that a group of samples (same label) is from all other samples in hierarchical clustering.

        This function calculates the distance from a certain number of samples with the same label to the closest samples using the 
        hierarchical clustering linkage matrix and the labels (in Spectra) of each sample. For each set of samples with the same label, it 
        calculates the difference of distances between where the cluster with all the set of samples was formed and the cluster that joins 
        those set of samples with another samples. The normalization of this distance is made by dividing said difference by the max 
        distance between any two cluster. If the samples with the same label aren't all in the same cluster before any other sample joins 
        them, the distance given to this set of samples is zero. It returns the measure of the normalized distance as well as a dictionary 
        with all the calculated distances for all set of samples (labels).

        df: Pandas DataFrame.
        Z: ndarray; hierarchical clustering linkage matrix (from scipy.cluster.hierarchical.linkage)
        method: str; Available methods - "average", "median". This is the method to give the normalized discrimination distance measure
        based on the distances calculated for each set of samples.

        Returns: (global_distance, discrimination_distances)
        global_distance: float or None; normalized discrimination distance measure
        discrimination_distances: dict: dictionary with the discrimination distance for each label.
    """

    # Get metadata from df
    unique_labels = df.cdl.unique_labels
    n_unique_labels = df.cdl.label_count
    all_labels = list(df.cdl.labels)
    ns = len(df.cdl.samples)

    # Create dictionary with number of samples per label
    sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}
    min_len = min(sample_number.values())
    max_len = max(sample_number.values())

    # to_tree() returns root ClusterNode and ClusterNode list
    _, cn_list = hier.to_tree(Z, rd=True)

    # print('results from to_tree ----------------')
    # for cn in cn_list[ns:]:
    #     n_in_cluster = cn.get_count()
    #     if not (min_len <= n_in_cluster <= max_len):
    #         continue
    #     ids = cn.pre_order(lambda x: x.id)
    #     labels = [all_labels[i] for i in ids]
    #     d = Z[cn.get_id() - ns, 2]
    #     print(f'{cn.get_id()} --> {n_in_cluster} --> {ids} -- > {labels}')
    #     print('distance = ', d)

    # print('-------------------------------------')

    # dists is a dicionary of cluster_id: distance. Distance is fetch from Z

    dists = {cn.get_id(): Z[cn.get_id() - ns, 2] for cn in cn_list[ns:]}

    # Calculate discriminating distances of a set of samples with the same label
    # store in dictionary `discrims`.
    # `total` accumulates total.

    total = 0
    discrims = {label: 0.0 for label in unique_labels}
    # Z[-1,2] is the maximum distance between any 2 clusters
    max_dist = Z[-1,2]

    for cn in cn_list:
        i = cn.get_id()
        len_cluster = cn.get_count()
        # skip if cluster too short or too long
        if not (min_len <= len_cluster <= max_len):
            continue

        labelset = [all_labels[loc] for loc in cn.pre_order(lambda x: x.id)]
        # get first element
        label0 = labelset[0]

        # check if cluster length == exactely the number of samples of label of 1st element.
        # If so, all labels must also be equal
        if len_cluster != sample_number[label0] or labelset.count(label0) != len_cluster:
            continue

        # Compute distances.
        # find iteration when cluster i was integrated in a larger one
        itera = np.where(Z == i)[0][0]
        dif_dist = Z[itera, 2]

        discrims[label0] = (dif_dist - dists[i]) / max_dist
        total += discrims[label0]
        # print(f'\n-----------\ncluster {i}, label set ----> {labelset}')
        # print('discrims ---->', discrims)
        # print('total so far ---->', total)

    # Method to quantify a measure of a global discriminating distance for a linkage matrix.
    if method == 'average':
        separaM = total / n_unique_labels
    elif method == 'median':
        separaM = np.median(list(discrims.values()))
        if separaM == 0:
            separaM = None
    else:
        raise ValueError('Method should be one of: ["average", "median"].')
    # print('return values *********')
    # print(separaM)
    # print(discrims)
    # print('************************')

    return separaM, discrims


# Hierarchical Clustering - function necessary for calculation of Baker's Gamma Correlation Coefficient
def mergerank(Z):
    """Creates a 'rank' of the iteration number two samples were linked to the same cluster.
       
       Z: 2-D array; the return of the linkage function in scypy.stats.hierarchy.

       Returns: Matrix/2-D array; Symmetrical Square Matrix (dimensions: len(Z)+1 by len(Z)+1), (i,j) position is the iteration 
    number sample i and j were linked to the same cluster (higher rank means the pair took more iterations to be linked together).
    """
    nZ = len(Z)
    kmatrix = np.zeros((nZ + 1, nZ + 1))
    # Creating initial cluster matrix
    clust = {}
    for i in range(0, nZ + 1):
        clust[i] = (float(i),)
    # Supplementing cluster dictionary with clusters as they are made in hierarchical clustering and filling matrix with the number of
    # the hierarchical clustering iteration where 2 samples were linked together.
    for r in range(0, nZ):
        if Z[r, 0] < nZ + 1 and Z[r, 1] < nZ + 1:
            kmatrix[int(Z[r, 0]), int(Z[r, 1])] = r + 1
            kmatrix[int(Z[r, 1]), int(Z[r, 0])] = r + 1
            # Dictionary with the elements in the cluster formed at iteration r. - r: (elements)
            clust[nZ + 1 + r] = (Z[r, 0], Z[r, 1], )
        else:
            # Dictionary with the elements in the cluster formed at iteration r.
            clust[nZ + 1 + r] = (clust[Z[r, 0]] + clust[Z[r, 1]])  
            for i in range(0, len(clust[Z[r, 0]])):
                for j in range(0, len(clust[Z[r, 1]])):
                    kmatrix[int(clust[Z[r, 0]][i]), int(clust[Z[r, 1]][j])] = r + 1
                    kmatrix[int(clust[Z[r, 1]][j]), int(clust[Z[r, 0]][i])] = r + 1
    return kmatrix


# K-means Clustering
# Discrimination Distance for k-means
def Kmeans_discrim(df, method='average'):
    """Gives a measure of the normalized distance that a group of samples (same label) is from all other samples in k-means clustering.

       This function performs a k-means clustering with the default parameters of sklearn.cluster.KMeans except the number of clusters
    (equal to the number of unique labels of the spectra). It then checks each of the clusters formed to see if only the samples of a
    label/group are present and if all of them are present. If a group doesn't fulfill the conditions, a distance of zero is given to
    that set of labels. If it fulfills them, the distance is calculated as the distance between the centroid of the samples cluster
    and the closest centroid. The distance is normalized by dividing it by the maximum distance between any 2 cluster centroids. It then
    returns the mean/median of the discrimination distances of all groups and a dictionary with each individual discrimination distance.

    df: Pandas DataFrame.
    method: str (default: "average"); Available methods - "average", "median". This is the method to give the
    normalized discrimination distance measure based on the distances calculated for each set of samples.

    Returns: (scalar, dictionary); discrimination distance measure, dictionary with the discrimination distance for each set of 
    samples.
    """
    # Application of the K-means clustering with n_clusters equal to the number of unique labels.

    # Get data parts
    # DataParts = namedtuple('DataParts', 'data_matrix labels names features unique_labels')
    dfdata = df.cdl.data
    unique_labels = list(dfdata.unique_labels)
    all_labels = list(dfdata.labels)
    n_labels = len(unique_labels)
    sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}

    Kmean2 = skclust.KMeans(n_clusters=n_labels)
    Kmean = Kmean2.fit(dfdata.data_matrix)

    Correct_Groupings = {label: 0 for label in unique_labels}

    # Creating dictionary with number of samples for each group

    # Making a matrix with the pairwise distances between any 2 clusters.
    distc = dist.pdist(Kmean.cluster_centers_)
    distma = dist.squareform(distc)
    maxi = max(distc)  # maximum distance (to normalize discrimination distancces).

    # Check if the two conditions are met (all samples in one cluster and only them)
    # Then calculate discrimination distance.
    for i in unique_labels:
        if (Kmean.labels_ == Kmean.labels_[all_labels.index(i)]).sum() == sample_number[
            i
        ]:
            Correct_Groupings[i] = (
                min(
                    distma[Kmean.labels_[all_labels.index(i)], :][distma[Kmean.labels_[all_labels.index(i)], :] != 0
                    ]
                )
                / maxi
            )

    # Method to quantify a measure of a global discriminating distance for k-means clustering.
    if method == 'average':
        Correct_Groupings_M = np.array(list(Correct_Groupings.values())).mean()
    elif method == 'median':
        Correct_Groupings_M = np.median(list(Correct_Groupings.values()))
        if Correct_Groupings_M == 0:
            Correct_Groupings_M = None
    else:
        raise ValueError(
            'Method not recognized. Available methods: "average", "median".'
        )

    return Correct_Groupings_M, Correct_Groupings


# SMOTE oversampling method
def fast_SMOTE(df, binary=False, max_sample=0):
    """Performs a fast oversampling of a set of spectra based on the simplest SMOTE method.

       New samples are artificially made using the formula: New_Sample = Sample1 + random_value * (Sample2 - Sample1), where the 
    random_value is a randomly generated number between 0 and 1. One new sample is made from any combinations of two different samples
    belonging to the same group/label.

    df: DataFrame.
    binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
    max_sample: integer (default: 0); number of maximum samples for each label. If < than the label with the most amount of
    samples, this parameter is ignored. Samples chosen to be added to the dataset are randomly selected from all combinations of
    two different samples belonging to the same group/label.

    Returns: DataFrame; Table with extra samples with the name 'Arti(Sample1)-(Sample2)'.
    """
    Newdata = df.copy().cdl.erase_labels()

    n_unique_labels = df.cdl.label_count
    unique_labels = df.cdl.unique_labels
    all_labels = list(df.cdl.labels)
    n_all_labels = len(all_labels)

    nlabels = []
    nnew = {}
    for i in range(n_unique_labels):
        # See how many samples there are in the dataset for each unique_label of the dataset
        # samples = [df.iloc[:,n] for n, x in enumerate(all_labels) if x == unique_labels[i]]
        label_samples = [df.cdl.subset(label=lbl) for lbl in unique_labels]
        if len(label_samples) > 1:
            # if len(samples) = 1 - no pair of 2 samples to make a new one.
            # Ensuring all combinations of samples are used to create new samples.
            n = len(label_samples) - 1
            for j in range(len(label_samples)):
                m = 0
                while j < n - m:
                    Vector = label_samples[n - m] - label_samples[j]
                    random = np.random.random(1)
                    if binary:
                        # Round values to 0 or 1 so the data stays binary while still creating "relevant" "new" data.
                        Newdata[
                            'Arti' + unique_labels[j] + '-' + unique_labels[n - m]
                        ] = round(label_samples[j] + random[0] * Vector)
                    else:
                        Newdata[
                            'Arti' + unique_labels[j] + '-' + unique_labels[n - m]
                        ] = (label_samples[j] + random[0] * Vector)
                    m = m + 1
                    # Giving the correct label to each new sample.
                    nlabels.append(unique_labels[i])

        # Number of samples added for each unique label
        if i == 0:
            nnew[unique_labels[i]] = len(nlabels)
        else:
            nnew[unique_labels[i]] = len(nlabels) - sum(nnew.values())

    # Creating dictionary with number of samples for each group
    sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}

    # Choosing samples for each group/labels to try and get max_samples in total of that label.
    if max_sample >= max(sample_number.values()):
        # List with names of the samples chosen for the final dataset
        chosen_samples = list(df.samples)
        nlabels = []
        loca = 0
        for i in unique_labels:
            # Number of samples to add
            n_choose = max_sample - sample_number[i]
            # If there aren't enough new samples to get the total max_samples, choose all of them.
            if n_choose > nnew[i]:
                n_choose = nnew[i]
            # Random choice of the samples for each label that will be added to the original dataset
            chosen_samples.extend(
                rd.sample(
                    list(
                        Newdata.columns[
                            n_all_labels + loca : n_all_labels + loca + nnew[i]
                        ]
                    ),
                    n_choose,
                )
            )
            loca = loca + nnew[i]
            nlabels.extend([i] * n_choose)

        # Creating the dataframe with the chosen samples
        Newdata = Newdata[chosen_samples]

    # Creating the label list for the AlignedSpectra object
    Newlabels = all_labels + nlabels
    Newdata = mtl.add_labels(Newdata, labels=Newlabels)
    return Newdata


# Random Forests Functions - simple_RF, RF_M3 (Method 3), RF_M4 (Method 4)

# simple_RF - RF application and result extraction.
def simple_RF(df, iter_num=20, n_fold=3, n_trees=200):
    """Performs k-fold cross validation on random forest classification of a dataset n times giving its accuracy and ordered most
    important features.

       Spectra: AlignedSpectra object (from metabolinks).
       iter_num: int (default - 20); number of iterations that random forests are repeated.
       n_fold: int (default - 3); number of groups to divide dataset in for k-fold cross-validation
            (max n_fold = minimum number of samples belonging to one group).
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scores, import_features); 
            scores: list of the scores/accuracies of k-fold cross-validation of the random forests
                (one score for each each iteration and each group)
            import_features: list of tuples (index number of feature, feature importance, feature name)
                ordered by decreasing feature importance.
    """

    # Get data parts
    # DataParts = namedtuple('DataParts', 'data_matrix labels names features unique_labels')
    dfdata = df.cdl.data
    all_labels = list(dfdata.labels)
    # n_labels = len(unique_labels)
    # sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}
    nfeats = len(df.index)

    # Setting up variables for result storing
    imp_feat = np.zeros((iter_num * n_fold, nfeats))
    cv = []
    f = 0
    for i in range(iter_num):  # number of times random forests cross-validation is made
        # Dividing dataset in balanced n_fold groups
        kf = StratifiedKFold(n_fold, shuffle=True)
        CV = []
        # Repeating for each of the n groups the random forest model fit and classification
        for train_index, test_index in kf.split(df.T, all_labels):
            # Random Forest setup and fit
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            X_train, X_test = (
                df[df.columns[train_index]].T,
                df[df.columns[test_index]].T,
            )
            y_train, y_test = (
                [all_labels[i] for i in train_index],
                [all_labels[i] for i in test_index],
            )
            rf.fit(X_train, y_train)

            # Obtaining results with the test group
            CV.append(rf.score(X_test, y_test))
            imp_feat[f, :] = rf.feature_importances_
            f = f + 1

        cv.append(np.mean(CV))
    
    # Join and order all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis=0) / (iter_num * n_fold)
    sorted_imp_feat = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_tuples = [(loc, importance, df.index[loc]) for loc, importance in sorted_imp_feat]

    return cv, imp_feat_tuples


# In disuse
# Function for method 3 - SMOTE on the training set
def RF_M3(df, iter_num=20, binary=False, test_size=0.1, n_trees=200):
    """Performs random forest classification of a dataset (oversampling the training set) n times giving its mean score, Kappa Cohen 
    score, most important features and cross-validation score.

       df: DataFrame.
       iter_num: int (default - 20); number of iterations that random forests are repeated.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scalar, scalar, list of tuples); mean of the scores of the random forests, mean of the Cohen's Kappa score of 
    the random forests, descending ordered list of tuples with index number of feature, feature importance and feature name.
    """
    # Get data parts
    # DataParts = namedtuple('DataParts', 'data_matrix labels names features unique_labels')
    dfdata = df.cdl.data
    all_labels = list(dfdata.labels)
    # n_labels = len(unique_labels)
    # sample_number = {label: len(df.cdl.samples_of(label)) for label in unique_labels}

    imp_feat = np.zeros((iter_num, len(df)))
    cks = []
    scores = []

    for i in range(iter_num):
        # Splitting data and performing SMOTE on the training set.
        X_train, X_test, y_train, y_test = train_test_split(
            df.T, all_labels, test_size=test_size
        )
        # X_Aligned = AlignedSpectra(X_train.T, labels = y_train)
        X_Aligned = X_train.T
        X_Aligned.cdl.labels = y_train
        Spectra_S = fast_SMOTE(X_Aligned, binary=binary)
        # Random Forest setup and fit.
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        rf.fit(Spectra_S.T, Spectra_S.cdl.labels)

        # Extracting the results of the random forest model built
        y_pred = rf.predict(X_test)
        imp_feat[i, :] = rf.feature_importances_
        cks.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test, y_test))

    # Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis=0) / iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_ord = []
    for i, j in imp_feat_sum:
        imp_feat_ord.append((i, j, df.index[i]))

    return np.mean(scores), np.mean(cks), imp_feat_ord


# In disuse
# Function for method 3 - SMOTE on the training set and NGP processing of training and test data together.
def RF_M4(df, reffeat, iter_num=20, test_size=0.1, n_trees=200):
    """Performs random forest classification of a dataset (after oversampling the training sets and data processing both sets) n times 
    giving its mean score, Kappa Cohen score, most important features and cross-validation score.

       df: DataFrame.
       reffeat: scalar; m/z of the reference feature to normalize the samples.
       iter_num: int (default - 20); number of iterations that random forests are repeated.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scalar, scalar, list of tuples); mean of the scores of the random forests, mean of the Cohen's Kappa score of 
    the random forests, descending ordered list of tuples with index number of feature, feature importance and feature name.
    """
    imp_feat = np.zeros((iter_num, len(df) - 1))
    accuracy = []
    scores = []
    for i in range(iter_num):
        # Splitting data and performing SMOTE on the training set.
        X_train, X_test, y_train, y_test = train_test_split(
            df.T, df.cdl.labels, test_size=test_size
        )
        X_Aligned = X_train.T
        Spectra_S = fast_SMOTE(X_Aligned, binary=False)

        # NGP processing of the data
        Spectra_S_J = Spectra_S.join(X_test.T)
        Spectra_S_J.labels = Spectra_S.labels + y_test
        Norm_S = trans.normalize_ref_feature(Spectra_S_J, reffeat)
        glog_S = trans.glog(Norm_S)
        Euc_glog_S = trans.pareto_scale(glog_S)
        X_train = Euc_glog_S.data.iloc[:, : -len(y_test)]
        X_test = Euc_glog_S.data.iloc[:, -len(y_test) :]

        # Random Forest setup and fit.
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        rf.fit(X_train.T, Euc_glog_S.labels[: -len(y_test)])

        # Extracting the results of the random forest model built
        y_pred = rf.predict(X_test.T)
        imp_feat[i, :] = rf.feature_importances_
        accuracy.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test.T, y_test))

    # Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis=0) / iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_ord = []
    for i, j in imp_feat_sum:
        imp_feat_ord.append((i, j, df.index[i]))

    return np.mean(scores), np.mean(accuracy), imp_feat_ord


# Test the data with the training data, then check the difference with simple_RF. If this one is much higher, there is clear overfitting
def overfit_RF(Spectra, iter_num=20, test_size=0.1, n_trees=200):
    """Performs random forest classification of a dataset n times giving its mean score, Kappa Cohen score, most important features and
    cross-validation score.

       Spectra: AlignedSpectra object (from metabolinks).
       iter_num: int (default - 20); number of iterations that random forests are repeated.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scalar, scalar, list of tuples, scalar); mean of the scores of the random forests, mean of the Cohen's Kappa score of 
    the random forests, descending ordered list of tuples with index number of feature, feature importance and feature name, mean of
    3-fold cross-validation score.
    """
    imp_feat = np.zeros((iter_num, len(Spectra)))
    cks = []
    scores = []
    CV = []

    for i in range(iter_num):  # number of times random forests are made
        # Random Forest setup and fit
        rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        # X_train, X_test, y_train, y_test = train_test_split(Spectra.T,
        # Spectra.cdl.labels, test_size = test_size)
        rf.fit(Spectra.T, Spectra.cdl.labels)

        # Extracting the results of the random forest model built
        y_pred = rf.predict(Spectra.T)
        imp_feat[i, :] = rf.feature_importances_
        cks.append(cohen_kappa_score(Spectra.cdl.labels, y_pred))
        scores.append(rf.score(Spectra.T, Spectra.cdl.labels))
        CV.append(np.mean(cross_val_score(rf, Spectra.T, Spectra.cdl.labels, cv=3)))

    # Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis=0) / iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)
    imp_feat_ord = []
    for i, j in imp_feat_sum:
        imp_feat_ord.append((i, j, Spectra.index[i]))

    return np.mean(scores), np.mean(cks), imp_feat_ord, np.mean(CV)


def permutation_RF(df, iter_num=100, n_fold=3, n_trees=200):
    """Performs permutation test n times with k-fold cross validation of a dataset for random forest classification giving its accuracy
    score for the original and all permutations made and respective p-value.

       df: DataFrame.
       iter_num: int (default - 100); number of permutations that will be made.
       n_fold: int (default - 3); number of groups to divide dataset in for k-fold cross-validation (max n_fold = minimum number of
    samples belonging to one group).
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scalar, list of scalars, scalar); accuracy of k-fold cross-validation of the random forests made, accuracy of all
    permutations made of k-fold cross-validation of the random forest made, p-value (number of times permutation accuracy > original
    accuracy + 1)/(number of permutations + 1).
    """
    # Setting up variables for result storing
    Perm = []
    # List of columns to shuffle and dataframe of the data to put columns in NewC shuffled order
    NewC = list(df.columns.copy())
    df = df.copy()
    dft = df.transpose()
    # For dividing the dataset in balanced n_fold groups with a random random state maintained in all permutations (identical splits)
    kf = StratifiedKFold(
        n_fold, shuffle=True, random_state=np.random.randint(1000000000)
    )
    all_labels = tuple(df.cdl.labels)

    for i in range(iter_num + 1):
        # number of different permutations + original dataset where random forests cross-validation will be made
        # Temporary dataframe with columns in order of the NewC
        temp = df[NewC]
        perm = []
        splits = kf.split(dft, all_labels)
        # Repeat for each of the k groups the random forest model fit and classification
        for train_index, test_index in splits:
            # Random Forest setup and fitting
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            # X_train, X_test = temp[temp.columns[train_index]].T, temp[temp.columns[test_index]].T
            X_train, X_test = temp.iloc[:, train_index].T, temp.iloc[:, test_index].T
            y_train, y_test = (
                [all_labels[pos] for pos in train_index],
                [all_labels[pos] for pos in test_index],
            )

            rf.fit(X_train, y_train)
            # Obtaining results with the test group
            perm.append(rf.score(X_test, y_test))

        # Shuffle dataset columns - 1 permutation of the columns (leads to permutation of labels)
        np.random.shuffle(NewC)

        # Appending k-fold cross-validation score
        Perm.append(np.mean(perm))

    # Taking out k-fold cross-validation accuracy for the non-shuffled (labels) dataset and p-value calculation
    CV = Perm[0]
    pvalue = (sum(Perm[1:] >= Perm[0]) + 1) / (iter_num + 1)

    return CV, Perm[1:], pvalue


# Function to calculate a correlation coefficient between 2 different hierarchical clusterings/ dendrograms.
def Dendrogram_Sim(Z, zdist, Y, ydist, type='cophenetic', Trace=False):
    """Calculates a correlation coefficient between 2 dendograms based on their distances and hierarchical clustering performed.
    
       Z: ndarray; linkage matrix of hierarchical clustering 1.
       zdist: ndarray; return of the distance function in scypy.spatial.distance for hierarchical clustering 1.
       Y: ndarray; linkage matrix of hierarchical clustering 2.
       ydist: ndarray; return of the distance function in scypy.spatial.distance for hierarchical clustering 2.
       simtype: string (default - 'cophenetic'); types of correlation coefficient metrics to use; accepted: {'Baker Kendall', 'Baker 
    Spearman', 'cophenetic'}.
       Trace: bool (default - False); gives a report of the correlation coefficient.
       
       Returns: (float, float); correlation coefficient of specified type and respective p-value.
    """

    if type == 'cophenetic':
        CophZ = hier.cophenet(Z, zdist)
        CophY = hier.cophenet(Y, ydist)
        r, p = stats.pearsonr(CophZ[1], CophY[1])
        if Trace:
            print(
                'The Cophenetic Correlation Coefficient is {} , and has a p-value of {}'.format(
                    r, p
                )
            )
        return (r, p)

    else:
        KZ = mergerank(Z)
        KY = mergerank(Y)
        SZ = KZ[KZ != 0]
        SY = KY[KY != 0]
        if type == 'Baker Kendall':
            Corr = stats.kendalltau(SZ, SY)
            if Trace:
                print(
                    'The Baker (Kendall) Correlation Coefficient is:',
                    Corr[0],
                    ', and has a p-value of',
                    Corr[1],
                )
            return Corr
        elif type == 'Baker Spearman':
            Corr = stats.spearmanr(SZ, SY)
            if Trace:
                print(
                    'The Baker (Spearman) Correlation Coefficient is:',
                    Corr[0],
                    ', and has a p-value of',
                    Corr[1],
                )
            return Corr
        else:
            raise ValueError(
                'Type not Recognized. Types accepted: "Baker Kendall", "Baker Spearman", "cophenetic"'
            )


# --------- PLS-DA functions ---------------------


def optim_PLS(df, matrix, max_comp=50, n_fold=3):
    """Searches for an optimum number of components to use in PLS-DA by accuracy (3-fold cross validation) and mean-squared errors.

    df: DataFrame; X equivalent in PLS-DA (training vectors).
    matrix: pandas DataFrame; y equivalent in PLS-DA (target vectors).
    max_comp: integer; upper limit for the number of components used.
    n_fold: int (default - 3); number of groups to divide dataset in
    for k-fold cross-validation (max n_fold = minimum number of samples
    belonging to one group).

    Returns: (list, list, list), 3-fold cross-validation score and r2 score and mean squared errors for all components searched.
    """
    # Preparating lists to store results
    CVs = []
    CVr2s = []
    MSEs = []
    # Repeating for each component from 1 to max_comp
    for i in range(1, max_comp + 1):
        cv = []
        cvr2 = []
        mse = []
        all_labels = list(df.cdl.labels)

        # Splitting data into 3 groups for 3-fold cross-validation
        kf = StratifiedKFold(n_fold, shuffle=True)
        # Repeating for each of the 3 groups
        for train_index, test_index in kf.split(df.T, all_labels):
            plsda = PLSRegression(n_components=i, scale=False)
            # X_train, X_test = df[df.columns[train_index]].T, df[df.columns[test_index]].T
            X_train, X_test = df.iloc[:, train_index].T, df.iloc[:, test_index].T
            y_train, y_test = (
                matrix.T[matrix.T.columns[train_index]].T,
                matrix.T[matrix.T.columns[test_index]].T,
            )

            # Fitting the model
            plsda.fit(X=X_train, Y=y_train)

            # Obtaining results with the test group
            cv.append(plsda.score(X_test, y_test))
            cvr2.append(r2_score(plsda.predict(X_test), y_test))
            y_pred = plsda.predict(X_test)
            mse.append(mean_squared_error(y_test, y_pred))
        # Storing results for each number of components
        CVs.append(np.mean(cv))
        CVr2s.append(np.mean(cvr2))
        MSEs.append(np.mean(mse))

    return CVs, CVr2s, MSEs


def _calculate_vips(model):
    """ VIP (Variable Importance in Projection) of the PLSDA model for each variable in the system.

        model: PLS Regression model fitted to a dataset from scikit-learn.

        returns: list; VIP score for each variable from the dataset.
    """
    # Set up the variables
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))

    # Calculate VIPs
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)

    return vips


def model_PLSDA(df, matrix, n_comp,
                n_fold=3,
                iter_num=100,
                feat_type='Coef',
                figures=False
):
    """Perform PLS-DA with 3-fold cross-validation.

       df: pandas DataFrame; includes X equivalent in PLS-DA (training vectors).
       matrix: pandas DataFrame; y equivalent in PLS-DA (target vectors).
       n_comp: integer; number of components to use in PLS-DA.
       n_fold: int (default: 3); number of groups to divide dataset in for k-fold cross-validation (max n_fold = minimum number of
    samples belonging to one group).
    iter_num: int (default: 100); number of iterations that PLS-DA is repeated.
       feat_type: string (default: 'Coef'); types of feature importance metrics to use; accepted: {'VIP', 'Coef', 'Weights'}.
       figures: bool/int (default: False); only for 3-fold CV, if an integer n, shows distribution of samples of n groups in 3 scatter
    plots with the 2 most important latent variables (components) - one for each group of cross-validation.

       Returns: (list, list, list, list of tuples, list of tuples); accuracy in group selection, 3-fold cross-validation score, r2 score
    of the model, descending ordered list of tuples with index number of feature, feature importance and name of feature for X_Weights 
    and Regression Coefficients (two methods of feature selection) respectively.
    """
    # Setting up lists and matrices to store results
    CV = []
    CVR2 = []
    Accuracy = []
    Imp_Feat = np.zeros((iter_num * n_fold, len(df.index)))
    f = 0
    all_labels = list(df.cdl.labels)

    # Number of iterations equal to iter_num
    for i in range(iter_num):
        # Splitting data into 3 groups for 3-fold cross-validation
        kf = StratifiedKFold(n_fold, shuffle=True)
        # Setting up variables for results of the application of 3-fold cross-validated PLS-DA
        certo = 0
        cv = []
        cvr2 = []

        # Repeating for each of the 3 groups
        for train_index, test_index in kf.split(df.T, all_labels):
            plsda = PLSRegression(n_components=n_comp, scale=False)
            X_train, X_test = df.iloc[:, train_index].T, df.iloc[:, test_index].T
            y_train, y_test = (
                matrix.T[matrix.T.columns[train_index]].T,
                matrix.T[matrix.T.columns[test_index]].T,
            )

            # Fit the model
            plsda.fit(X=X_train, Y=y_train)

            # Obtain results with the test group
            cv.append(plsda.score(X_test, y_test))
            cvr2.append(r2_score(plsda.predict(X_test), y_test))
            y_pred = plsda.predict(X_test)

            # Decision to which group each sample belongs to based on y_pred
            # Decision rule chosen: sample belongs to group where it has max y_pred (closer to 1)
            for i in range(len(y_pred)):
                # if list(y_test.iloc[:,i]).index(max(y_test.iloc[:,i])) == np.argmax(y_pred[i]):
                if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                    y_pred[i]
                ):
                    certo = certo + 1  # Correct prediction

            # Calculate important features (3 different methods to choose from)
            if feat_type == 'VIP':
                Imp_Feat[f, :] = _calculate_vips(plsda)
            elif feat_type == 'Coef':
                Imp_Feat[f, :] = abs(plsda.coef_).sum(axis=1)
            elif feat_type == 'Weights':
                Imp_Feat[f, :] = abs(plsda.x_weights_).sum(axis=1)
            else:
                raise ValueError(
                    'Type not Recognized. Types accepted: "VIP", "Coef", "Weights"'
                )

            f += 1

            # figures = True - making scatter plots of training data in the 2 first components
            LV_score = pd.DataFrame(plsda.x_scores_)

            # TODO: this should be moved to another function (separation of computation from plots)
            if figures != False:
                # Preparing colours to separate different groups
                colours = cm.get_cmap('nipy_spectral', figures)
                col_lbl = colours(range(figures))
                col_lbl = list(col_lbl)
                for i in range(len(col_lbl)):
                    a = 2 * i
                    col_lbl.insert(a + 1, col_lbl[a])

                # Scatter plot
                ax = LV_score.iloc[:, 0:2].plot(
                    x=0, y=1, kind='scatter', s=50, alpha=0.7, c=col_lbl, figsize=(9, 9)
                )
                # Labeling each point
                i = -1
                for n, x in enumerate(LV_score.values):
                    if n % 2 == 0:
                        i = i + 1
                    label = df.cdl.unique_labels[i]
                    # label = LV_score.index.values[n]
                    ax.text(x[0], x[1], label, fontsize=8)

        # Calculate the accuracy of the group predicted and storing score results
        Accuracy.append(certo / len(all_labels))
        CV.append(np.mean(cv))
        CVR2.append(np.mean(cvr2))

    # Join and sort all important features values from each cross validation group and iteration.
    Imp_sum = Imp_Feat.sum(axis=0) / (iter_num * n_fold)
    Imp_sum = sorted(enumerate(Imp_sum), key=lambda x: x[1], reverse=True)
    Imp_ord = []
    for i, j in Imp_sum:
        Imp_ord.append((i, j, df.index[i]))

    return Accuracy, CV, CVR2, Imp_ord


def permutation_PLSDA(df, n_comp, n_fold=3, iter_num=100, figures=False):
    """Perform permutation test of PLS-DA on an AlignedSpectra with 3-fold cross-validation used to obtain the model's and its
    permutations accuracy.

       df: DataFrame. Includes X and Y equivalent in PLS-DA (training vectors and groups).
       n_comp: integer; number of components to use in PLS-DA.
       n_fold: int (default - 3); number of groups to divide dataset in for k-fold cross-validation (max n_fold = minimum number of
    samples belonging to one group).
    iter_num: int (default - 100); number of permutations made (times labels are shuffled).

       Returns: (scalar, list of scalars, scalar); accuracy of k-fold cross-validation of the PLS-DA made, accuracy of all
    permutations made of k-fold cross-validation of the PLS-DA made, p-value (number of times permutation accuracy > original
    accuracy + 1)/(number of permutations + 1).
    """
    # Setting up a list to store results
    Accuracy = []

    # Setting list of columns to shuffle and dataframe of the data to put columns in NewC shuffled order
    NewC = list(df.columns.copy())
    df = df.copy()

    # Matrix formation
    all_labels = list(df.cdl.labels)
    unique_labels = list(df.cdl.unique_labels)

    if len(unique_labels) > 2:
        # Setting up the y matrix for when there are more than 2 classes (multi-class)
        matrix = pd.get_dummies(all_labels)
        matrix = matrix[unique_labels]
    else:
        # Setting the y list when there are only 2 classes
        matrix = unique_labels

    # Splitting data into n_fold groups for n-fold cross-validation
    kf = StratifiedKFold(
        n_fold, shuffle=True, random_state=np.random.randint(1000000000)
    )
    # Number of permutations + dataset with non-shuffled labels equal to iter_num + 1
    for i in range(iter_num + 1):
        # Temporary dataframe with columns in order of the NewC
        temp = df[NewC]
        # Setting up variables for results of the application of 3-fold cross-validated PLS-DA
        certo = 0

        # Repeating for each of the n groups
        for train_index, test_index in kf.split(df.T, all_labels):
            # plsda model building for each of the n stratified groups amde
            plsda = PLSRegression(n_components=n_comp, scale=False)
            X_train, X_test = (
                temp[temp.columns[train_index]].T,
                temp[temp.columns[test_index]].T,
            )
            y_train, y_test = (
                matrix.T[matrix.T.columns[train_index]].T,
                matrix.T[matrix.T.columns[test_index]].T,
            )
            # Fitting the model
            plsda.fit(X=X_train, Y=y_train)

            # Obtaining results with the test group
            y_pred = plsda.predict(X_test)

            # Decision to which group each sample belongs to based on y_pred
            # Decision rule chosen: sample belongs to group where it has max y_pred (closer to 1)
            for i in range(len(y_pred)):
                if list(y_test.iloc[i, :]).index(max(y_test.iloc[i, :])) == np.argmax(
                    y_pred[i]
                ):
                    certo = certo + 1  # Correct prediction

        # Calculating the accuracy of the group predicted and storing score results (for original and permutated labels)
        Accuracy.append(certo / len(all_labels))
        # Shuffle dataset labels - 1 permutation of the labels
        np.random.shuffle(NewC)

    # Taking k-fold cross-validation accuracy for the non-shuffled (labels) dataset and p-value calculation
    CV = Accuracy[0]
    pvalue = (
        sum(
            [Accuracy[i] for i in range(1, len(Accuracy)) if Accuracy[i] >= Accuracy[0]]
        )
        + 1
    ) / (iter_num + 1)

    return CV, Accuracy[1:], pvalue
