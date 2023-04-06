from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import mne
import numpy as np
import vg
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit


def fit_correlations(
    x_corr: np.ndarray, y_corr: np.ndarray, fit_function: Callable, fit_diag: bool = True
) -> np.ndarray:
    """fit x and y with a given fit_function, if a 2d array is passed, it is flattened
    Returns
    fit_params: np.ndarray
        Array with fitted parameters
    """

    remove_idx = 0 if fit_diag else x_corr.shape[0]
    sort_idx = np.argsort(x_corr, axis=None)
    x_sorted = x_corr.flatten()[sort_idx]
    y_sorted = y_corr.flatten()[sort_idx]
    fit_params, _ = curve_fit(fit_function, x_sorted[remove_idx:], y_sorted[remove_idx:])
    return fit_params


def get_corr_and_var(cov_matrix: np.ndarray) -> tuple(np.ndarray, np.ndarray):
    """Returns correlation matrix and variances computed from a covariance matrix
    Note that the variance is returned as a diagonal matrix"""
    variance = np.diag(np.diag(cov_matrix))
    stdev = np.diag(np.sqrt(variance))
    # like in the implementation of np.corrcoef
    corr_matrix = (
        cov_matrix / stdev[:, None] / stdev[None, :]
    )  # divided by column and row vector of Standard deviations
    return corr_matrix, variance


def get_channel_distance_matrix(
    montage: mne.channels.DigMontage | None = None,
    mne_info: mne.Info | None = None,
    distance_metric: Literal["3d", "angle", "geodesic"] = "3d",
) -> np.ndarray:
    """Compute a NxN symmetric matrix containing all the pairwise electrode distances
    corresponding to the covariance"""
    metric_list = ["3d", "angle", "geodesic"]
    if distance_metric not in metric_list:
        raise ValueError(f"{distance_metric=} Possible metrics: {metric_list}")
    ch_pos = montage.get_positions()["ch_pos"]
    n_chans = len(ch_pos)
    mat = np.zeros((n_chans, n_chans))
    for c in range(n_chans):
        cur_ch_name = montage.ch_names[c]
        cur_ch_pos = ch_pos[cur_ch_name]
        if distance_metric == "3d":
            x = np.array([np.linalg.norm(cur_ch_pos - ch_pos[k]) for k in ch_pos])
        elif distance_metric == "angle":
            _, head_center, _ = mne.bem.fit_sphere_to_headshape(mne_info)
            x = np.array(
                [vg.angle(cur_ch_pos - head_center, ch_pos[k] - head_center) for k in ch_pos]
            )
        elif distance_metric == "geodesic":
            radius, head_center, _ = mne.bem.fit_sphere_to_headshape(mne_info)
            angles = np.array(
                [vg.angle(cur_ch_pos - head_center, ch_pos[k] - head_center) for k in ch_pos]
            )
            x = (np.pi * radius * angles) / 180
        mat[:, c] = x
    return mat


def get_spatial_model(
    template_cov: np.ndarray,
    fit_function: Callable[..., np.ndarray],
    montage: mne.channels.DigMontage | None = None,
    mne_info: mne.Info | None = None,
    distance_metric: Literal["3d", "angle", "geodesic"] = "3d",
    model_variances: np.ndarray | None = None,
    model_fit_diag: bool = False,
    return_corr: bool = False,
) -> np.ndarray:
    """Construct a spatial model of template_cov by fitting the correlations.
    Returns either a model correlation or covariance matrix. When returning
    a covariance matrix, the variances can be passed via model_variances
     or the template variances are used
     Options
    model_variances: np.ndarray
        Can be used when returning a covariance model to specify the variances.
        Should be a diagonal matrix.
    model_fit_diag: bool
        When True, the diagonal value of the correlation matrix are not used for fitting,
        as they are 1 by definition. Can influence accuracy and conditioning of the model
    return_corr: bool
        Wether to return a correlation or covariance matrix model

    """
    if not isinstance(montage, mne.channels.DigMontage):
        raise ValueError(
            f"montage needs to be an instance of {mne.channels.DigMontage}. Current type: {type(montage)}"
        )
    if distance_metric != "3d" and not isinstance(mne_info, mne.Info):
        raise ValueError(
            f"mne_info needs to be an instance of {mne.Info} to compute angle or geodesic distances."
        )

    distance_matrix = get_channel_distance_matrix(
        montage=montage, mne_info=mne_info, distance_metric=distance_metric
    )
    template_corr, template_var = get_corr_and_var(template_cov)
    # If its a partial function and its only remaining argument is x (i.e. all parameters are fixed)
    if (
        isinstance(fit_function, partial)
        and len(set(fit_function.func.__code__.co_varnames) - set(fit_function.keywords.keys()))
        == 1
    ):
        model_corr = fit_function(distance_matrix)
    else:
        fit_params = fit_correlations(
            distance_matrix, template_corr, fit_function, fit_diag=model_fit_diag
        )
        model_corr = fit_function(distance_matrix, *fit_params)
        np.fill_diagonal(model_corr, 1)
    if return_corr:
        return model_corr
    else:
        if model_variances is None:
            model_variances = template_var
        return sqrtm(model_variances) @ model_corr @ sqrtm(model_variances)
