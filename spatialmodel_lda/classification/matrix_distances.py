# Different ways to compute matrix distances
import numpy as np


def nmse(true_mat: np.ndarray, esti_mat: np.ndarray) -> float:
    """calculates the squared Frobenius norm, squared distance between 2 matrices"""
    return np.linalg.norm(true_mat - esti_mat) / (np.linalg.norm(true_mat))


def log_euclidan(true_mat: np.ndarray, esti_mat: np.ndarray):
    raise NotImplementedError("implemented in pyRiemann")


def riemann_distance(true_mat: np.ndarray, esti_mat: np.ndarray):
    raise NotImplementedError("implemented in pyRiemann")
