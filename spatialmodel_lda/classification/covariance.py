# Created by chris at 16.03.23 00:58
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def subtract_classwise_means(xTr, y, ext_mean=None):
    n_classes = len(np.unique(y))
    n_features = xTr.shape[0]
    X = np.zeros((n_features, 0))
    cl_mean = np.zeros((n_features, n_classes))
    for ci, cur_class in enumerate(np.unique(y)):
        class_idxs = y == cur_class
        cl_mean[:, ci] = np.mean(xTr[:, class_idxs], axis=1)

        if ext_mean is None:
            X = np.concatenate(
                [
                    X,
                    xTr[:, class_idxs]
                    - np.dot(cl_mean[:, ci].reshape(-1, 1), np.ones((1, np.sum(class_idxs)))),
                ],
                axis=1,
            )
        else:
            X = np.concatenate(
                [
                    X,
                    xTr[:, class_idxs]
                    - np.dot(ext_mean[:, ci].reshape(-1, 1), np.ones((1, np.sum(class_idxs)))),
                ],
                axis=1,
            )
    return X, cl_mean


def calc_n_times(dim, n_channels, n_times):
    if type(n_times) is int:
        return n_times
    elif n_times == "infer":
        if dim % n_channels != 0:
            raise ValueError("Could not infer time samples, Remainder is nonzero")
        else:
            return dim // n_channels
    else:
        raise ValueError(f"Unknown value for n_times: {n_times}")


def calc_scm(
    x_0: np.ndarray,
    x_1: Optional[np.ndarray] = None,
    return_means: bool = False,
):
    """Computes the Sample or the Cross Covariance Matrix"""
    if x_1 is None:
        x_1 = x_0
    p, n = x_0.shape
    mu_0 = np.repeat(np.mean(x_0, axis=1, keepdims=True), n, axis=1)
    mu_1 = np.repeat(np.mean(x_1, axis=1, keepdims=True), n, axis=1)
    Xn_0 = x_0 - mu_0
    Xn_1 = x_1 - mu_1
    S = np.matmul(Xn_0, Xn_1.T)
    Cstar = S / (n - 1)
    if not return_means:
        return Cstar
    else:
        return Cstar, (mu_0, mu_1)


def shrinkage(
    X: np.ndarray,
    gamma: Optional[float] = None,
    T: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    standardize: bool = True,
) -> Tuple[np.ndarray, float]:

    """
    Computes a shrunk covariance matrix
    Ported from the BBCI Matlab implementation of shrinkage
    as described in Schäfer & Strimmer 2005.

    Keep in Mind that when passing T  without standardization it
    needs to be multiplied by the N-1 the way it is implemented right now.
    With standardization its basically shrinking the correlations
    """

    # case for gamma = auto (ledoit-wolf)
    p, n = X.shape

    if standardize:
        sc = StandardScaler()  # standardize features
        X = sc.fit_transform(X.T).T
    Xn = X - np.repeat(np.mean(X, axis=1, keepdims=True), n, axis=1)
    if S is None:
        S = np.matmul(Xn, Xn.T)
    Xn2 = np.square(Xn)
    idxdiag = np.diag_indices(p)

    nu = np.mean(S[idxdiag])
    if T is None:
        T = nu * np.eye(p, p)

    # Ledoit Wolf
    if gamma is None:
        V = 1.0 / (n - 1) * (np.matmul(Xn2, Xn2.T) - np.square(S) / n)
        gamma = n * np.sum(V) / np.sum(np.square(S - T))
    if gamma > 1:
        print("logger.warning('forcing gamma to 1')")
        gamma = 1
    elif gamma < 0:
        print("logger.warning('forcing gamma to 0')")
        gamma = 0
    Cstar = (gamma * T + (1 - gamma) * S) / (n - 1)
    if standardize:  # scale back
        Cstar = sc.scale_[np.newaxis, :] * Cstar * sc.scale_[:, np.newaxis]

    return Cstar, gamma


def shrink_to_cond(
    S,
    T=None,
    target_cond=2000,
    start_gamma=0.3,
    gamma_rate=0.005,
    precision_thresh=50,
    max_iters=1000,
):
    """Evaluates condition number of matrix shrunk with decreasing gammas until target condition is reached.
    Returns shrunk matrix and condition number"""
    iters = 0
    current_gamma = start_gamma
    cond = np.linalg.cond(S)
    cond_diff = cond - target_cond  #
    if T is None:
        T = np.diag(np.diag(S))
    while abs(cond_diff) > precision_thresh and iters < max_iters:
        # For now the rate is fixed, could include gradient for better accuracy.
        current_gamma -= gamma_rate
        shrunk_S = (1 - current_gamma) * S + current_gamma * T
        cond = np.linalg.cond(shrunk_S)
        cond_diff = cond - target_cond
        # print(cond)
        iters += 1
        if cond_diff > 0:
            break
        if current_gamma - gamma_rate < 0:
            # print(f"reached lowest gamma: {gamma_rate}")
            break
    return shrunk_S, current_gamma
