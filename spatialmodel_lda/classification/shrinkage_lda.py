# Created by chris at 16.03.23 00:15
import numpy as np
import sklearn
from pyriemann.utils.distance import distance
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearClassifierMixin
from sklearn.utils.validation import check_X_y

from spatialmodel_lda.classification.covariance import (
    calc_n_times,
    shrink_to_cond,
    shrinkage,
    subtract_classwise_means,
)
from spatialmodel_lda.classification.matrix_distances import nmse


class ShrinkageLinearDiscriminantAnalysis(
    BaseEstimator,
    LinearClassifierMixin,
):
    """SLDA Implementation with lots of options, including the option to include 'oracle data' in the fit function
    to explore theoretical scenarios and assess the impact of incorporating additional information like resting data
    into the covariance estimate. It also includes Info params, a dict with diagnostic information about the shrunk and
     estimated matrices."""

    def __init__(
        self,
        priors: np.ndarray | None = None,
        n_times: str | int = "infer",
        n_channels: int = 31,
        pool_cov: bool = True,
        standardize_shrink: bool = True,
        unit_w: bool = False,
        only_block: bool = False,
        scm_gamma: float = None,
        oracle_gamma: float = None,
        calc_oracle_mean: bool = False,
        calc_oracle_cov: bool = False,
        use_oracle_cov: bool = False,
        fixed_scm_cond: float | None = None,
        fixed_scm_cond_start: float | None = 0.3,
        fixed_scm_cond_rate: float | None = 0.002,
        return_info_params: bool = True,
    ):
        if fixed_scm_cond is not None and scm_gamma != 0:
            raise ValueError("Only shrinking to a fixed cond if the SCM is not shrunk already!")
        self.priors = priors
        self.n_times = n_times
        self.n_channels = n_channels
        self.pool_cov = pool_cov
        self.standardize_shrink = standardize_shrink
        self.unit_w = unit_w
        self.only_block = only_block
        self.scm_gamma = scm_gamma
        self.oracle_gamma = oracle_gamma
        self.calc_oracle_mean = calc_oracle_mean
        self.calc_oracle_cov = calc_oracle_cov
        self.use_oracle_cov = use_oracle_cov
        self.fixed_scm_cond = fixed_scm_cond
        self.fixed_scm_cond_start = fixed_scm_cond_start
        self.fixed_scm_cond_rate = fixed_scm_cond_rate
        self.return_info_params = return_info_params
        if self.return_info_params:
            self.info_params = dict()

    def fit(self, X_train, y, montage=None, info=None, oracle_data=None):
        # Section: Basic setup
        check_X_y(X_train, y)
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError("currently only binary class supported")
        assert len(X_train) == len(y)
        xTr = X_train.T

        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
        else:
            priors = self.priors

        # Section: covariance / mean calculation
        X, cl_mean = subtract_classwise_means(xTr, y)
        if self.pool_cov:
            C_cov, self.scm_gamma = shrinkage(
                X,
                standardize=self.standardize_shrink,
                gamma=self.scm_gamma,
            )
        else:
            C_cov = np.zeros((xTr.shape[0], xTr.shape[0]))
            for cur_class in self.classes_:
                class_idxs = y == cur_class
                x_slice = X[:, class_idxs]
                C_cov += priors[cur_class] * shrinkage(x_slice)[0]

        if self.fixed_scm_cond is not None:
            C_cov, self.scm_gamma = shrink_to_cond(
                C_cov,
                target_cond=self.fixed_scm_cond,
                start_gamma=self.fixed_scm_cond_start,
                gamma_rate=self.fixed_scm_cond_rate,
            )

        if self.calc_oracle_mean:
            _, oracle_means = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
            cl_mean = oracle_means
        if self.calc_oracle_cov:
            oracle_X, oracle_means = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
            oracle_cov, self.oracle_gamma = shrinkage(
                oracle_X, gamma=self.oracle_gamma, standardize=self.standardize_shrink
            )
            if self.use_oracle_cov:
                C_cov = oracle_cov
        else:
            oracle_cov = None

        # Only use Diagonal Blocks (Spatial covariances) and set everything else to zero
        if self.only_block:
            self.n_times = calc_n_times(X.shape[0], self.n_channels, self.n_times)
            C_cov_new = np.zeros_like(C_cov)
            for i in range(self.n_times):
                idx_start = i * self.n_channels
                idx_end = idx_start + self.n_channels
                block = C_cov[idx_start:idx_end, idx_start:idx_end]
                C_cov_new[idx_start:idx_end, idx_start:idx_end] = block
            C_cov = C_cov_new

        if self.return_info_params:
            cov_matrices = {"sample_cov": C_cov}
            self._calc_info_params(cov_matrices, oracle_cov)

        w = np.linalg.solve(C_cov, cl_mean)
        w = w / np.linalg.norm(w) if self.unit_w else w
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + np.log(priors)

        if len(self.classes_) == 2:
            w = w[:, 1] - w[:, 0]
            b = b[1] - b[0]

        self.coef_ = w.reshape((1, -1))
        self.intercept_ = b

    def _calc_info_params(self, cov_matrices, oracle_cov=None):
        self.info_params["scm_gamma"] = round(self.scm_gamma, 4)
        if self.calc_oracle_cov:
            self.info_params["oracle_gamma"] = round(self.oracle_gamma, 4)
        for key, matrix in cov_matrices.items():
            self.info_params[key + "_cond"] = np.round(np.linalg.cond(matrix), 2)
            if self.calc_oracle_cov:
                self.info_params[key + "_nmse_dist"] = round(nmse(oracle_cov, matrix), 4)
                try:
                    self.info_params[key + "_logeuc_dist"] = round(
                        distance(oracle_cov, matrix, metric="logeuclid"), 4
                    )
                except ValueError:
                    self.info_params[key + "_logeuc_dist"] = np.nan

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))
