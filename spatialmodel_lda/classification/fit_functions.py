from functools import partial

import numpy as np

"""Collection of functions to model Correlation matrices."""


def angle_fit_func(r, beta):
    """
    Function to model spatial correlations between EEG electrodes.
    Adapted from Beltrachini et al., 2013.
    Added a normalization term because without it go above 1 and below 0, which is usually not seen.
    r : radius of the spherical shell containing the dipoles Unit in meters
    beta: angle between the ith and jth electrodes w.r.t. the center of the sphere IN RADIANS
    """
    numerator = (1 - r**4) * ((1 - (2 * np.cos(beta) * (r**2)) + r**4) ** (-3 / 2)) - 1
    denominator = (r**2) * (3 - r) * 1 / (1 - r**2)
    old_res = numerator / denominator
    normalized = (old_res - np.min(old_res)) / (np.max(old_res) - np.min(old_res))
    return normalized


# These are the new basic functions.
def exp_func(x: np.ndarray, b, k, s):
    return 2 - b * np.exp((k * x) ** s)


def exp_func_huizenga(x: np.ndarray, alpha, beta):
    return np.exp(-((x / alpha) ** beta))


def poly2_func(x: np.ndarray, a, b, c):
    return a * x + b * x**2 + c


def straight_func_free(x: np.ndarray, a, b):
    return a * x + b


def straight_func1(x: np.ndarray, a):
    "assured to go through the origin"
    b = 1
    return a * x + b


def exp_func_bfree_s1(x: np.ndarray, b, k):
    return 2 - b * np.exp(k * x)


def exp_func_bfree_s2(x: np.ndarray, b, k):
    return 2 - b * np.exp((k * x) ** 2)


def exp_func_b1_s1(x: np.ndarray, k):
    "assured to go through the origin"
    b = 1
    return 2 - b * np.exp(k * x)


def exp_func_b1_s2(x: np.ndarray, k):
    "assured to go through the origin"
    b = 1
    return 2 - b * np.exp((k * x) ** 2)


def exp_func_bsfree(x: np.ndarray, b, k, s):
    return 2 - b * np.exp((k * x) ** s)


def exp_func_b1_sfree(x: np.ndarray, k, s):
    "assured to go through the origin"
    b = 1
    return 2 - b * np.exp((k * x) ** s)


def exp_func_huizenga_s1(x: np.ndarray, alpha):
    return np.exp(-((x / alpha)))


def exp_func_huizenga_s2(x: np.ndarray, alpha):
    return np.exp(-((x / alpha) ** 2))


def poly2_func_cfree(x: np.ndarray, a, b, c):
    return a * x + b * x**2 + c


def poly2_func_c1(x: np.ndarray, a, b):
    "assured to go through the origin"
    c = 1
    return a * x + b * x**2 + c


def cos_func(x: np.ndarray, w):

    return np.cos(w * x)
