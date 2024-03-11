# !usr/bin/env python
# Created by "shyammarjit", modified by "A. Zubov" -%
#       Email:
#             DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import math
import numpy as np
from scipy import sparse
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def MICC(X, y, cutoff, w=0.9):

    n_cutoff = int(math.floor(cutoff*(X.shape[1])))

    if sparse.issparse(X):
        feat_corr = np.abs(corrcoef_sparse(X))
    else:
        # memory-efficient calculation
        if X.shape[0] > 10000 or X.shape[1] > 10000:
            feat_corr = np.abs(corrcoef_memmap(X.values))
        else:
            feat_corr = np.abs(np.corrcoef(X.values, rowvar=False))
    feat_pcc = np.zeros(len(feat_corr))
    for i in range(0, len(feat_corr)):
        # subtract 1 to exclude the autocorrelation
        feat_pcc[i] = (np.sum(feat_corr[i]) - 1)/(len(feat_corr) - 1)
    
    if y.dtype == "float":
        if sparse.issparse(X):
            mi = mutual_info_regression(X, y.values)
        else:
            mi = mutual_info_regression(X.values, y.values)
    else:
        if sparse.issparse(X):
            mi = mutual_info_classif(X, y.values)
        else:
            mi = mutual_info_classif(X.values, y.values)
    
    mi_pcc = w*mi - (1 - w)*feat_pcc
    mi_pcc_idxed = np.concatenate(
        (mi_pcc[:,  np.newaxis], np.arange(0, len(feat_corr))[:, np.newaxis]), axis=1)
    
    feat_mi_pcc_indexes = np.lexsort((mi_pcc_idxed[:, 1], mi_pcc_idxed[:, 0]))
    mi_pcc_idxed[feat_mi_pcc_indexes][::-1]

    mi_pcc_selected = mi_pcc_idxed[:n_cutoff]
    feat_selected_indexes = mi_pcc_selected[:, 1].astype(int)

    return feat_selected_indexes

def corrcoef_sparse(matrix):
    n, m = matrix.shape
    means = matrix.sum(axis=0)/n

    centered_mat = matrix - np.outer(np.ones(n), means)
    C = centered_mat.T @ centered_mat

    sqrt_diag_inv = 1.0/np.sqrt(np.diag(C))
    sqrt_diag_inv[np.isinf(sqrt_diag_inv)] = 0
    sqrt_diag_inv_mat = sparse.csc_matrix(np.diag(sqrt_diag_inv))

    corrcoefs = sqrt_diag_inv_mat @ C @ sqrt_diag_inv_mat
    np.fill_diagonal(corrcoefs, 1.0)

    return corrcoefs

def corrcoef_memmap(matrix):
    z0 = matrix - np.mean(matrix, 0)
    sigma = np.std(matrix, 0)

    N = matrix.shape[1]
    corrcoefs = np.memmap("corrcoef_memmap.npy", dtype="float16", mode="w+", shape=(N, N))
    np.dot(z0.T.astype("float16"), z0.astype("float16"), out=corrcoefs)
    corrcoefs /= sigma[None, :]
    corrcoefs /= sigma[:, None]

    return corrcoefs