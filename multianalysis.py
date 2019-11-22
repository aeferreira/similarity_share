import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier 
import scipy.stats as stats
import scaling as sca
import sklearn.cluster as skclust
import sklearn.ensemble as skensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
from metabolinks import AlignedSpectra

#Apart from mergerank, other functions are incredibly specific to the objectives of this set of notebooks.
#Functions present are for the different kinds of multivariate analysis made in the Jupyter Notebooks

#Hierarchical Clustering - function necessary for calculation of Baker's Gamma Correlation Coefficient
def mergerank(Z):
    """Creates a 'rank' of the iteration number two samples were linked to the same cluster.
       
       Z: 2-D array; the return of the linkage function in scypy.stats.hierarchy.

       Returns: Matrix/2-D array; Symmetrical Square Matrix (dimensions: len(Z)+1 by len(Z)+1), (i,j) position is the iteration 
    number sample i and j were linked to the same cluster (higher rank means the pair took more iterations to be linked together).
    """
    kmatrix = np.zeros((len(Z)+1, len(Z)+1))
    #Creating initial cluster matrix
    clust = {}
    for i in range(0,len(Z)+1):
        clust[i] = (float(i),)
    #Supplementing cluster dictionary with clusters as they are made in hierarchical clustering and filling matrix with the number of 
    #the hierarchical clustering iteration where 2 samples were linked together.
    for r in range(0,len(Z)):
        if Z[r,0] < len(Z)+1 and Z[r,1] < len(Z)+1:
            kmatrix[int(Z[r,0]),int(Z[r,1])] = r+1
            kmatrix[int(Z[r,1]),int(Z[r,0])] = r+1
            clust[len(Z)+1+r] = Z[r,0],Z[r,1] #Dictionary with the elements in the cluster formed at iteration r. - r: (elements)
        else:
            clust[len(Z)+1+r] = clust[Z[r,0]] + clust[Z[r,1]] #Dictionary with the elements in the cluster formed at iteration r.
            for i in range(0,len(clust[Z[r,0]])):
                for j in range(0,len(clust[Z[r,1]])):
                    kmatrix[int(clust[Z[r,0]][i]),int(clust[Z[r,1]][j])] = r+1
                    kmatrix[int(clust[Z[r,1]][j]),int(clust[Z[r,0]][i])] = r+1
    return kmatrix

#K-means Clustering
#Discrimination Distance for k-means
def Kmeans_discrim(Spectra, sample_number, method = 'average'):
    """Gives a measure of the normalized distance that a group of samples (same label) is from all other samples in k-means clustering.

       This function performs a k-means clustering with the default parameters of sklearn.cluster.KMeans except the number of clusters 
    that is equal to the number of unique labels of the spectra. It then checks each of the clusters formed to see if only the samples 
    of a label/group are present and if all of them are present. If a group doesn't fulfill the conditions, a distance of zero is given 
    to that set of labels. If it fulfills them, the distance is calculated as the distance between the centroid of the samples cluster 
    and the closest centroid. The distance is normalized by dividing it by the maximum distance between any 2 cluster centroids. It then
    returns the mean/median of the discrimination distances of all groups amd a dictionary with each individual discrimination distance.

       Spectra: AlignedSpectra object (from metabolinks).
       sample_number: int; number of samples of a set (all with the same label in Spectra).
       method: str; Available methods - "average", "median". This is the method to give the normalized discrimination distance measure
    based on the distances calculated for each set of samples.

       Returns: (scalar, dictionary); discrimination distance measure, dictionary with the discrimination distance for each set of 
    samples.
    """
    #Application of the K-means clustering with n_clusters equal to the number of unique labels.
    Kmean2 = skclust.KMeans(n_clusters = len(Spectra.unique_labels()))
    Kmean = Kmean2.fit(Spectra.data.T)
    Clusters = {}
    Correct_Groupings = dict(zip(Spectra.unique_labels(), [
                      0] * len(Spectra.unique_labels())))

    #Making a matrix with the pairwise distances between any 2 clusters.                      
    distc = dist.pdist(Kmean.cluster_centers_) 
    distma = dist.squareform(distc)
    maxi = max(distc) #maximum distance (to normalize discrimination distacnces).
    #Check if the two conditions are met (all samples in one cluster and only them) and calculation of discrimination distance.
    for i in range(0,len(Spectra.labels), sample_number):
        Clusters[Spectra.labels[i]] = Kmean.labels_[i:i+sample_number]
        if (Kmean.labels_ == Kmean.labels_[i]).sum() == sample_number:
            Correct_Groupings[Spectra.labels[i]] = min(distma[Kmean.labels_[i],:][distma[Kmean.labels_[i],:]!=0])/maxi

    #Method to quantify a measure of a global discriminating distance for k-means clustering.            
    if method == 'average':
        Correct_Groupings_M = np.array(list(Correct_Groupings.values())).mean()
    elif method == 'median':
        Correct_Groupings_M = np.median(list(Correct_Groupings.values()))
        if Correct_Groupings_M == 0:
            Correct_Groupings_M = None
    else:
        raise ValueError(
            'Method not recognized. Available methods: "average", "median".')
        
    return Correct_Groupings_M, Correct_Groupings


#SMOTE oversampling method
def fast_SMOTE(Spectra, binary = False):
    """Performs a fast oversampling of a set of spectra based on the simplest SMOTE method.

       New samples are artificially made using the formula: New_Sample = Sample1 + random_value * (Sample2 - Sample1), where the 
    random_value is a randomly generated number between 0 and 1. One new sample is made from any combinations of two different samples
    belonging to the same group (label).

       Spectra: AlignedSpectra object (from metabolinks).
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.

       Returns: AlignedSpectra object (from metabolinks); Spectra with extra samples originated with the name 'Arti(Sample1)-(Sample2)'.
    """
    Newdata = Spectra.data.copy()  
    nlabels = []
    for i in range(len(Spectra.unique_labels())):
        #See how many samples there are in the dataset for each unique_label of the dataset
        samples = [Spectra.data.iloc[:,n] for n, x in enumerate(Spectra.labels) if x == Spectra.unique_labels()[i]]
        if len(samples)>1: #if len(samples) = 1 - no pair of 2 samples to make a new one.
            #Ensuring all combinations of samples are used to create new samples.
            n = len(samples) - 1
            for j in range(len(samples)):
                m = 0
                while j < n - m:
                    Vector = samples[n - m] - samples[j]
                    random = np.random.random(1)
                    if binary:
                        #Round values to 0 or 1 so the data stays binary while still creating "relevant" "new" data.
                        Newdata['Arti' + samples[j].name + '-' + samples[n-m].name] = round(samples[j] + random[0]*Vector)
                    else:
                        Newdata['Arti' + samples[j].name + '-' + samples[n-m].name] = samples[j] + random[0]*Vector
                    m = m + 1
                    #Giving the correct label to each new sample.
                    nlabels.append(Spectra.unique_labels()[i]) 
    Newlabels = Spectra.labels + nlabels
    return AlignedSpectra(Newdata, labels = Newlabels)

#Random Forests Functions - simple_RF, RF_M3 (Method 3), RF_M4 (Method 4)

#simple_RF - RF application and result extraction.
def simple_RF(Spectra, iter_num = 20, test_size = 0.1, n_trees = 200):
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
    imp_feat = np.zeros((iter_num, len(Spectra.data)))
    cks = []
    scores = []
    CV = []
    
    for i in range(iter_num): #number of times random forests are made
        #Random Forest setup and fit
        rf = skensemble.RandomForestClassifier(n_estimators = n_trees)
        X_train, X_test, y_train, y_test = train_test_split(Spectra.data.T, 
                                                            Spectra.labels, test_size = test_size)
        rf.fit(X_train, y_train)
        
        #Extracting the results of the random forest model built
        y_pred = rf.predict(X_test)
        imp_feat[i,:] = rf.feature_importances_
        cks.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test, y_test))
        CV.append(np.mean(cross_val_score(rf, Spectra.data.T, Spectra.labels, cv=3)))
    
    #Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis = 0)/iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key = lambda x: x[1], reverse = True)
    imp_feat_ord = []
    for i,j in imp_feat_sum:
        imp_feat_ord.append((i , j, Spectra.data.index[i]))
    
    return np.mean(scores), np.mean(cks), imp_feat_ord, np.mean(CV)

#Function for method 3 - SMOTE on the training set
def RF_M3(Spectra, iter_num = 20, binary = False, test_size = 0.1, n_trees = 200):
    """Performs random forest classification of a dataset (oversampling the training set) n times giving its mean score, Kappa Cohen 
    score, most important features and cross-validation score.

       Spectra: AlignedSpectra object (from metabolinks).
       iter_num: int (default - 20); number of iterations that random forests are repeated.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scalar, scalar, list of tuples); mean of the scores of the random forests, mean of the Cohen's Kappa score of 
    the random forests, descending ordered list of tuples with index number of feature, feature importance and feature name.
    """
    imp_feat = np.zeros((iter_num, len(Spectra.data)))
    cks = []
    scores = []

    for i in range(iter_num):
        #Splitting data and performing SMOTE on the training set.
        X_train, X_test, y_train, y_test = train_test_split(Spectra.data.T, Spectra.labels, test_size=test_size)
        X_Aligned = AlignedSpectra(X_train.T, labels = y_train)
        Spectra_S = fast_SMOTE(X_Aligned, binary = binary)
        #Random Forest setup and fit.
        rf = skensemble.RandomForestClassifier(n_estimators = n_trees)
        rf.fit(Spectra_S.data.T, Spectra_S.labels)
        
        #Extracting the results of the random forest model built
        y_pred = rf.predict(X_test)
        imp_feat[i,:] = rf.feature_importances_
        cks.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test, y_test))

    #Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis = 0)/iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key = lambda x: x[1], reverse = True)
    imp_feat_ord = []
    for i,j in imp_feat_sum:
        imp_feat_ord.append((i , j, Spectra.data.index[i]))

    return np.mean(scores), np.mean(cks), imp_feat_ord

#Function for method 3 - SMOTE on the training set and NGP processing of training and test data together.
def RF_M4(Spectra, reffeat, iter_num = 20, test_size = 0.1, n_trees = 200):
    """Performs random forest classification of a dataset (after oversampling the training sets and data processing both sets) n times 
    giving its mean score, Kappa Cohen score, most important features and cross-validation score.

       Spectra: AlignedSpectra object (from metabolinks).
       reffeat: scalar; m/z of the reference feature to normalize the samples.
       iter_num: int (default - 20); number of iterations that random forests are repeated.
       binary: bool (default - False); indication if the Spectra has binary data and therefore also ensuring the new samples made are
    also binary or if the Spectra has a "normal" non-binary dataset.
       test_size: scalar (default - 0.1); number between 0 and 1 equivalent to the fraction of the samples for the test group.
       n_trees: int (default - 200); number of trees in each random forest.

       Returns: (scalar, scalar, list of tuples); mean of the scores of the random forests, mean of the Cohen's Kappa score of 
    the random forests, descending ordered list of tuples with index number of feature, feature importance and feature name.
    """    
    imp_feat = np.zeros((iter_num, len(Spectra.data)-1))
    accuracy = []
    scores = []
    for i in range(iter_num):
        #Splitting data and performing SMOTE on the training set.
        X_train, X_test, y_train, y_test = train_test_split(Spectra.data.T, Spectra.labels, test_size = test_size)
        X_Aligned = AlignedSpectra(X_train.T, labels = y_train)
        Spectra_S = fast_SMOTE(X_Aligned, binary = False)
        
        #NGP processing of the data
        Spectra_S_J = Spectra_S.data.join(X_test.T)
        Spectra_S_J = AlignedSpectra(Spectra_S_J, labels = Spectra_S.labels + y_test)
        Norm_S = sca.Norm_Feat(Spectra_S_J, reffeat)
        glog_S = sca.glog(Norm_S, 0)
        Euc_glog_S = sca.ParetoScal(glog_S)
        X_train = Euc_glog_S.data.iloc[:,:-len(y_test)]
        X_test = Euc_glog_S.data.iloc[:,-len(y_test):]
        
        #Random Forest setup and fit.
        rf = skensemble.RandomForestClassifier(n_estimators = n_trees)
        rf.fit(X_train.T, Euc_glog_S.labels[:-len(y_test)])
        
        #Extracting the results of the random forest model built
        y_pred = rf.predict(X_test.T)
        imp_feat[i,:] = rf.feature_importances_
        accuracy.append(cohen_kappa_score(y_test, y_pred))
        scores.append(rf.score(X_test.T, y_test))
    
    #Joining and ordering all important features values from each random forest
    imp_feat_sum = imp_feat.sum(axis = 0)/iter_num
    imp_feat_sum = sorted(enumerate(imp_feat_sum), key = lambda x: x[1], reverse = True)
    imp_feat_ord = []
    for i,j in imp_feat_sum:
        imp_feat_ord.append((i , j, Spectra.data.index[i]))
    
    return np.mean(scores), np.mean(accuracy), imp_feat_ord

#Function to calculate a correlation coefficient between 2 different hierarchical clusterings/ dendrograms.
def Dendrogram_Sim(Z, zdist, Y, ydist, type = 'cophenetic', Trace = False):
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
        Corr = stats.pearsonr(CophZ[1],CophY[1])
        if Trace:
            print ('The Cophenetic Correlation Coefficient is:', Corr[ 0], ', and has a p-value of', Corr[1])
        return Corr

    else:
        KZ = mergerank (Z)
        KY = mergerank (Y)
        SZ = KZ[KZ!=0]
        SY = KY[KY!=0]
        if type == 'Baker Kendall': 
            Corr = stats.kendalltau(SZ,SY)
            if Trace:
                print ('The Baker (Kendall) Correlation Coefficient is:', Corr[0], ', and has a p-value of', Corr[1])
            return Corr
        elif type =='Baker Spearman':
            Corr = stats.spearmanr(SZ,SY)
            if Trace:
                print ('The Baker (Spearman) Correlation Coefficient is:', Corr[0], ', and has a p-value of', Corr[1])
            return Corr
        else:
            raise ValueError ('Type not Recognized. Types accepted: "Baker Kendall", "Baker Spearman", "cophenetic"')