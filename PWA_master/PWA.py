#coding:utf-8 允许中文注释
import numpy as np
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

def L2_normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)

def PWA(X,a=2,b=2):
    # select_load ='../data/filter_select/select_num_oxford.npy'
    select_load = '../data/filter_select/select_num_paris.npy'
    select_num = np.load(select_load)
    select_num_map=select_num[0:25]
    X=np.array(X)
    if X.shape[0]==1 :#some feature is saved as four dim
        X=X[0]
    aggregated_feature=[]
    # loop all part detectors
    for i, x in enumerate(X):
        # whether select this part detector
        if i in select_num_map:
            # norm
            sum = (x ** a).sum() ** (1. / a)
            if sum != 0:   # 防止分母为零
                weight=(x / sum) ** (1. / b)
            else:
                weight = x

            # weighted sum-polling
            aggregated_feature_part=weight*X
            aggregated_feature_part=aggregated_feature_part.sum(axis=(1, 2))
            aggregated_feature_part_normal=aggregated_feature_part

            # concatenation
            if aggregated_feature==[]:
                aggregated_feature=aggregated_feature_part_normal
            else:
                aggregated_feature=np.row_stack((aggregated_feature,aggregated_feature_part_normal))

    aggregated_feature = aggregated_feature.ravel()
    # norm
    aggregated_feature_normal = L2_normalize(np.array(aggregated_feature), copy=False)
    aggregated_feature_normal=aggregated_feature_normal.reshape((1,-1))
    return aggregated_feature_normal

def run_feature_processing_pipeline(features=None, d=128, whiten=True, copy=False, params=None):
    """
    Given a set of feature vectors, process them with PCA/whitening and return the transformed features.
    If the params argument is not provided, the transformation is fitted to the data.

    :param ndarray features:
        image features for transformation with samples on the rows and features on the columns
    :param int d:
        dimension of final features
    :param bool whiten:
        flag to indicate whether features should be whitened
    :param bool copy:
        flag to indicate whether features should be copied for transformed in place
    :param dict params:
        a dict of transformation parameters; if present they will be used to transform the features

    :returns ndarray: transformed features
    :returns dict: transform parameters
    """
    # Normalize
    if features != None:
        features = L2_normalize(features, copy=copy)

    # PCA Whiten and reduce dimension
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca = PCA(n_components=d, whiten=whiten, copy=False)
        features = pca.fit_transform(features)
        params = { 'pca': pca }

    # Normalize
    features = L2_normalize(features, copy=copy)

    return features, params


