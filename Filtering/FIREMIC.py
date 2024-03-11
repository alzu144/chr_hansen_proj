# !usr/bin/env python
# Created by "Alexander Zubov", 23/10/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import math
import numpy as np
from sklearn.feature_selection import (mutual_info_regression, 
                                       mutual_info_classif, r_regression)


def FIREMIC(X, y, cutoff=0.1, dummy_size=0.1):
    """
    Feature importance using random elements, mutual information, and correlation

    Parameters
    ----------
    X : pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : pandas.Series of shape (n_samples,)
        Target vector.

    Returns
    -------
    feat_selected_indexes : numpy.ndarray
        Indices of the features selected.

    """
    n_feats = X.shape[1]
    n_cutoff = int(math.floor(cutoff*(X.shape[1])))

    n_dummies = int(n_feats * dummy_size)
    dummy_feats = np.random.rand(X.shape[0], n_dummies)

    if y.dtype == "float":
        mi_dummy = mutual_info_regression(dummy_feats, y.values)
        mi_X = mutual_info_regression(X.values, y.values)
        pr_dummy = r_regression(dummy_feats, y.values)
        pr_X = r_regression(X.values, y.values)
    else:
        mi_dummy = mutual_info_classif(dummy_feats, y.values)
        mi_X = mutual_info_classif(X.values, y.values)
        # pearson R work-in-progress

    mipr_indexed = np.concatenate((
        mi_X[:, None], pr_X[:, None], np.arange(0, n_feats)[:, None]
    ), axis=1)
    mipr_indexed_sorted = np.lexsort((mipr_indexed[:, 2], mipr_indexed[:, 0]))
    mipr_indexed = mipr_indexed[mipr_indexed_sorted][::-1]
    
    mi_thresholded = mipr_indexed[:, 0] > mi_dummy.max()
    pr_thresholded = np.abs(mipr_indexed[:, 1]) > np.abs(pr_dummy).max()
    threshold_indexes = np.logical_and(mi_thresholded, pr_thresholded)

    select_idx_tmp = np.where(threshold_indexes, mipr_indexed[:, 2], np.nan)

    feat_selected_indexes = select_idx_tmp[~np.isnan(select_idx_tmp)].astype(int)
    

    if len(feat_selected_indexes) > n_cutoff:
        feat_selected_indexes = feat_selected_indexes[:n_cutoff]

    return feat_selected_indexes