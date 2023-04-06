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
def straight_func(x: np.ndarray, a, c):
    return a * x + c


def poly2_func(x: np.ndarray, a, b, c):
    return a * x + b * x**2 + c


def exp_func(x: np.ndarray, b, k, s):
    return 2 - b * np.exp((k * x) ** s)


def exp_func_huizenga(x: np.ndarray, alpha, beta):
    "Spatial model function from Huizenga et al., 2002"
    return np.exp(-((x / alpha) ** beta))


def cos_func(x: np.ndarray, w):
    return np.cos(w * x)


# Modifications of basic functions with fixed parameters
# Wouldve preferred to use partial functions but curve_fit needs to able to inspect the func args
def straight_func1(x: np.ndarray, a):
    "assured to go through the origin"
    c = 1
    return a * x + c


def poly2_func_c1(x: np.ndarray, a, b):
    "assured to go through the origin"
    c = 1
    return a * x + b * x**2 + c


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


fit_functions = {
    "poly2-cfree": poly2_func,
    "exp1-bsfree": exp_func,
    "straight_func1": straight_func1,
    "poly2-c1": poly2_func_c1,
    "exp1-bfree-s1": exp_func_bfree_s1,
    "exp_func_bfree_s2": exp_func_bfree_s2,
    "exp1-b1-s1": exp_func_b1_s1,
    "exp1-b1-s2": exp_func_b1_s2,
    "exp1-b1-sfree": exp_func_b1_sfree,
    "exp2_s1": exp_func_huizenga_s1,
    "exp2_s2": exp_func_huizenga_s2,
}
