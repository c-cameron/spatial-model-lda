from __future__ import annotations

from functools import partial
from typing import Callable

import mne
import numpy as np
import vg
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit


def fit_correlations(
    x_corr: np.ndarray, y_corr: np.ndarray, fit_function: Callable, fit_diag: bool = True
):
    """fit x and y with a given fit_function, if a 2d array is passed, it is flattened
    COULD BE RENAMED SINCE IT CAN ACTUALY FIT ANYTHING
    """

    remove_idx = 0 if fit_diag else x_corr.shape[0]
    sort_idx = np.argsort(x_corr, axis=None)
    x_sorted = x_corr.flatten()[sort_idx]
    y_sorted = y_corr.flatten()[sort_idx]
    fit_params, _ = curve_fit(fit_function, x_sorted[remove_idx:], y_sorted[remove_idx:])
    return fit_params, x_sorted, y_sorted


def get_correlations(cov_matrix: np.ndarray):
    variance = np.diag(np.diag(cov_matrix))
    stdev = np.diag(np.sqrt(variance))
    # like in the implementation of np.corrcoef
    corr_matrix = (
        cov_matrix / stdev[:, None] / stdev[None, :]
    )  # divided by column and row vector of stdevs
    return variance, corr_matrix


def get_channel_distance_matrix(
    montage: mne.channels.DigMontage = None, mne_info: mne.Info = None, distance_metric: str = "3d"
) -> np.ndarray:
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
    montage: mne.channels.DigMontage = None,
    mne_info: mne.Info = None,
    distance_metric: str = "3d",
    model_variances: np.ndarray = None,
    model_fit_diag: bool = False,
    return_corr=False,
) -> np.ndarray:
    if not isinstance(montage, mne.channels.DigMontage):
        raise ValueError(f"montage is of type {type(montage)} not {mne.channels.DigMontage}")
    if distance_metric != "3d" and not isinstance(mne_info, mne.Info):
        raise ValueError(
            f"If distance metric is not '3d', mne_info needs to be an instance of {mne.Info}"
        )

    distance_matrix = get_channel_distance_matrix(
        montage=montage, mne_info=mne_info, distance_metric=distance_metric
    )
    template_var, template_corr = get_correlations(template_cov)
    # If its a partial function and its only remaining argument is x (i.e. all parameters are fixed)
    if (
        isinstance(fit_function, partial)
        and len(set(fit_function.func.__code__.co_varnames) - set(fit_function.keywords.keys()))
        == 1
    ):
        model_corr = fit_function(distance_matrix)
    else:
        fit_params, distance_sorted, corr_sorted = fit_correlations(
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
