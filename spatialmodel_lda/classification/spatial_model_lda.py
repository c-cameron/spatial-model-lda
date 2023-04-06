from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import sklearn.utils.multiclass
from pyriemann.utils.distance import distance
from scipy.linalg import sqrtm
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

import spatialmodel_lda.classification.covariance_modeling as cm
import spatialmodel_lda.classification.fit_functions as ff
from spatialmodel_lda.classification.covariance import (
    calc_n_times,
    calc_scm,
    shrinkage,
    subtract_classwise_means,
)
from spatialmodel_lda.classification.matrix_distances import nmse
from spatialmodel_lda.classification.shrinkage_lda import (
    ShrinkageLinearDiscriminantAnalysis,
)


class SpatialOnlyLda(ShrinkageLinearDiscriminantAnalysis):
    """
    Shrinkage LDA implementation using only the Spatial component of the EEG covariance matrix,
     i.e. the Diagonal Blocks. Also includes the option to perform shrinkage against a covariance
     model and many options to choose the type of model and shrinkage.


    Parameters
    ----------

    """

    def __init__(
        self,
        priors: np.ndarray | None = None,
        n_times: str | int = "infer",
        n_channels: int = 31,
        standardize_shrink: bool = True,
        scm_gamma: float | None = None,
        oracle_gamma: float | None = None,
        calc_oracle_mean: bool = False,
        calc_oracle_cov: bool = False,
        use_oracle_cov: bool = False,
        model_corr_source: Literal["resting", "decoupled", "oracle"] = "decoupled",
        model_var_source: Literal["decoupled", "oracle", "scn"] = "scm",
        distance_metric: Literal["3d", "angle", "geodesic"] = "3d",
        spatial_fit_function: Callable = ff.poly2_func_c1,
        model_fit_diag: bool = False,
        model_gamma: float = 0,
        model_oracle_gamma: float | None = None,
        fixed_scm_cond: float | None = None,  # Set a fixed condition number to reach by shrinkage
        fixed_scm_cond_start: float = 0.3,
        fixed_scm_cond_rate: float = 0.005,
        fixed_avg_cond: float | None = None,
        fixed_avg_cond_start: float = 0.8,
        fixed_avg_cond_rate: float = 0.005,
        return_info_params: bool = True,
    ):
        # SCM can be shrunk by either LW or by shrinking it to a certain condition number
        if fixed_scm_cond is not None and scm_gamma != 0:
            raise ValueError(
                "Only shrinking to a fixed condition number if the SCM is not shrunk already."
            )
        if fixed_avg_cond is not None and model_gamma is not None:
            raise ValueError(
                "Either model_gamma or fixed_avg_cond can be set at one time, choose one!"
            )
        self.priors = priors
        self.n_times = n_times
        self.n_channels = n_channels
        self.standardize_shrink = standardize_shrink
        self.scm_gamma = scm_gamma
        self.oracle_gamma = oracle_gamma
        self.calc_oracle_mean = calc_oracle_mean
        self.calc_oracle_cov = calc_oracle_cov
        self.use_oracle_cov = use_oracle_cov
        self.model_corr_source = model_corr_source
        self.model_var_source = model_var_source
        self.distance_metric = distance_metric
        self.spatial_fit_function = spatial_fit_function
        self.model_fit_diag = model_fit_diag
        self.model_gamma = model_gamma
        self.model_oracle_gamma = model_oracle_gamma
        self.fixed_scm_cond = fixed_scm_cond
        self.fixed_scm_cond_start = fixed_scm_cond_start
        self.fixed_scm_cond_rate = fixed_scm_cond_rate
        self.fixed_avg_cond = fixed_avg_cond
        self.fixed_avg_cond_start = fixed_avg_cond_start
        self.fixed_avg_cond_rate = fixed_avg_cond_rate
        self.return_info_params = return_info_params
        if self.return_info_params:
            self.info_params = {}
            self.param_list_dict = {
                "scm_gamma": [],
                "oracle_gamma": [],
                "model_gamma": [],
            }
            self.statfuncs = {
                "mean": np.mean,
                "std": np.std,
                "min": np.min,
                "median": np.median,
                "max": np.max,
            }
        # Flags to pass additional data during evaluations via hasattr check
        # Could also be implemented by inspecting function arguments or just try/except
        self.use_spatial_info = True
        self.use_resting_data = True

    def fit(
        self,
        X_train,
        y,
        montage=None,
        info=None,
        oracle_data=None,
        resting_data=None,
        resting_info=None,
    ):
        # Section: Basic setup
        check_X_y(X_train, y)
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        self.n_times = calc_n_times(X_train.shape[0], self.n_channels, self.n_times)
        if set(self.classes_) != {0, 1}:
            raise ValueError("currently only binary class supported")
        assert len(X_train) == len(y)
        xTr = X_train.T

        if self.priors is None:
            _, y_counts = np.unique(y, return_counts=True)
            priors = y_counts / float(len(y))
        else:
            priors = self.priors

        # Section: covariance / mean calculation
        X, class_means = subtract_classwise_means(xTr, y)
        # Computing the spatial covariance matrix for every timepoint, e.g. the diagonal blocks
        sample_cov_list = []
        for ii in range(self.n_times):
            block_X = X[ii * self.n_channels : self.n_channels + ii * self.n_channels, :]
            # Computing the Scm and shrinking to a fixed condition number
            if self.fixed_scm_cond is not None:
                block_cov = calc_scm(block_X)
                shrunk_block_cov, block_gamma = cm.shrink_to_cond(
                    block_cov,
                    target_cond=self.fixed_scm_cond,
                    start_gamma=self.fixed_scm_cond_start,
                    gamma_rate=self.fixed_scm_cond_rate,
                )
            # Computing the scm and shrinking to a specific gamma or via LW
            else:
                shrunk_block_cov, block_gamma = shrinkage(
                    block_X,
                    standardize=True,
                    gamma=self.scm_gamma,
                )
            sample_cov_list.append(shrunk_block_cov)
            if self.return_info_params:
                if self.scm_gamma is None or self.fixed_scm_cond is not None:
                    self.param_list_dict["scm_gamma"].append(round(block_gamma, 4))
        if self.return_info_params and len(self.param_list_dict["scm_gamma"]) == 0:
            self.param_list_dict["scm_gamma"] = self.scm_gamma

        if self.calc_oracle_mean:
            _, class_means = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
        if self.calc_oracle_cov:
            oracle_cov_list = list()
            oracle_X, oracle_means = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
            for ii in range(self.n_times):
                oracle_block_X = oracle_X[
                    ii * self.n_channels : self.n_channels + ii * self.n_channels,
                    :,
                ]
                oracle_block_cov, oracle_gamma = shrinkage(
                    oracle_block_X,
                    standardize=True,
                    gamma=self.oracle_gamma,
                )
                oracle_cov_list.append(oracle_block_cov)
                if self.return_info_params:
                    self.param_list_dict["oracle_gamma"].append(round(oracle_gamma, 4))
            if self.use_oracle_cov:
                sample_cov_list = oracle_cov_list
        else:
            oracle_cov_list = None

        # Section: Constructing the model
        if self.model_gamma == 0:
            avg_cov_list = sample_cov_list
        else:
            if self.model_var_source == "scm":
                model_variances = [np.diag(np.diag(cov)) for cov in sample_cov_list]
            if self.model_corr_source == "scm":
                raise NotImplementedError(
                    "Only using decoupled or oracle data to model correlations"
                )
            elif self.model_corr_source == "decoupled" or self.model_var_source == "decoupled":
                x_spatial = X.reshape((self.n_channels, -1), order="F")
                # Assuming that the data has more than enough samples that LW shrinkage works best
                template_cov, _ = shrinkage(
                    x_spatial,
                    gamma=None,
                )
                if self.model_var_source == "decoupled":
                    model_variances = np.diag(np.diag(template_cov))

            elif self.model_corr_source == "oracle" or self.model_var_source == "oracle":
                oracle_X, oracle_means = subtract_classwise_means(
                    oracle_data["x"].T, oracle_data["y"]
                )
                oracle_x_spatial = oracle_X.reshape((self.n_channels, -1), order="F")
                template_cov, oracle_spatial_gamma = shrinkage(
                    oracle_x_spatial,
                    gamma=self.model_oracle_gamma,
                )
                if self.model_var_source == "oracle":
                    model_variances = np.diag(np.diag(template_cov))
            elif self.model_corr_source == "resting":
                template_cov = calc_scm(resting_data)

            model_corr = cm.get_spatial_model(
                template_cov=template_cov,
                montage=montage,
                mne_info=info,
                fit_function=self.spatial_fit_function,
                distance_metric=self.distance_metric,
                model_fit_diag=self.model_fit_diag,
                return_corr=True,
            )

            if isinstance(model_variances, list):
                model_cov_list = [
                    sqrtm(variance) @ model_corr @ sqrtm(variance) for variance in model_variances
                ]
            else:
                model_cov = sqrtm(model_variances) @ model_corr @ sqrtm(model_variances)
                model_cov_list = [model_cov for _ in range(self.n_times)]
            # Shrinking the SCM towards the model target
            if self.fixed_avg_cond is not None:
                avg_cov_list = list()
                for ii, (block_cov, model_cov) in enumerate(zip(sample_cov_list, model_cov_list)):
                    shrunk_block_cov, model_cond_gamma = cm.shrink_to_cond(
                        S=block_cov,
                        T=model_cov,
                        target_cond=self.fixed_avg_cond,
                        start_gamma=self.fixed_avg_cond_start,
                        gamma_rate=self.fixed_avg_cond_rate,
                    )
                    avg_cov_list.append(shrunk_block_cov)
                    if self.return_info_params:
                        self.param_list_dict["model_gamma"].append(model_cond_gamma)
            else:
                avg_cov_list = [
                    self.model_gamma * model_cov + (1 - self.model_gamma) * sample_cov
                    for model_cov, sample_cov in zip(model_cov_list, sample_cov_list)
                ]
                if self.return_info_params:
                    self.param_list_dict["model_gamma"] = self.model_gamma

        if self.return_info_params:
            cov_matrices = {
                "sample_cov": sample_cov_list,
                "avg_cov": avg_cov_list,
            }
            if self.model_gamma != 0:
                cov_matrices["model_cov"] = model_cov_list
            self._calc_info_params(cov_matrices, oracle_cov_list)

        w_arr = np.zeros((self.n_channels, self.n_times))
        b_arr = np.zeros((1, self.n_times))

        # Computing weights and bias for every timepoint
        for i in range(self.n_times):
            start_idx = i * self.n_channels
            end_idx = start_idx + self.n_channels

            class_mean_time = class_means[start_idx:end_idx, :]
            w = np.linalg.solve(avg_cov_list[i], class_mean_time)

            b = -0.5 * np.sum(class_mean_time * w, axis=0).T + np.log(priors)
            w_arr[:, i] = w[:, 1] - w[:, 0]  # TODO unit w

            b_arr[:, i] = b[1] - b[0]

        self.coef_ = w_arr
        self.intercept_ = b_arr

    def _calc_info_params(self, cov_matrices: dict, oracle_cov_list: list | None = None):
        for key, val in cov_matrices.items():
            condlist = list()
            for matrix in val:
                condlist.append(np.round(np.linalg.cond(matrix), 4))
            self.param_list_dict[key + "_cond"] = condlist
            if self.calc_oracle_cov:
                self.param_list_dict[key + "_nmse_dist"] = [
                    round(nmse(oracle_cov, val_cov), 4)
                    for oracle_cov, val_cov in zip(oracle_cov_list, val)
                ]
                try:
                    self.param_list_dict[key + "_logeuc_dist"] = [
                        round(distance(oracle_cov, val_cov, metric="logeuclid"), 4)
                        for oracle_cov, val_cov in zip(oracle_cov_list, val)
                    ]
                except ValueError:
                    print("either matrix was not positive definite")
                    self.param_list_dict[
                        key + "_logeuc_dist"
                    ] = np.nan  # [np.nan for _ in oracle_cov_list]

        for gtype, params in self.param_list_dict.items():
            if not params:
                for key, func in self.statfuncs.items():
                    self.info_params[f"{gtype}_{key}"] = np.nan
            else:
                for key, func in self.statfuncs.items():
                    self.info_params[f"{gtype}_{key}"] = func(params)

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        from sklearn.utils.extmath import safe_sparse_dot

        check_is_fitted(self)
        self._validate_data(X, accept_sparse="csr", reset=False)
        n_features = self.coef_.shape[1]
        scores_arr = np.zeros((X.shape[0], self.n_times))
        for i in range(n_features):
            start_idx = i * self.n_channels
            end_idx = start_idx + self.n_channels
            scores_arr[:, i] = (
                safe_sparse_dot(X[:, start_idx:end_idx], self.coef_[:, i].T) + self.intercept_[:, i]
            )
        scores = np.mean(scores_arr, axis=1)
        """ The feature vector of each sample is n_channels * n_timepoints long, and
        since for spatial LDa we are basically computing an LDA for each timepoint separetely
        our coefficients are in the shape of n_channels * n_timepoints. we compute the score
        for each timepoint and then average them, using for loop as its more explicit"""
        # Alternative way with reshaping, note that we need to divide coefficients
        # coef = self.coef_.reshape(X.shape[1],order="F") / self.n_times
        # intercept = self.intercept_.mean()
        # scores = safe_sparse_dot(X, coef.T,
        #                          dense_output=True) + intercept
        return scores
