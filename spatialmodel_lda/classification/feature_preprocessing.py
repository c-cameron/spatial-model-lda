# Created by chris at 24.03.23 13:53
import mne
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EpochsVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        permute_channels_and_time=True,
        select_ival=None,
        jumping_mean_ivals=None,
        averaging_samples=None,
        rescale_to_uv=True,
        mne_scaler=None,
        pool_times=False,
        to_numpy_only=False,
        copy=True,
    ):
        self.permute_channels_and_time = permute_channels_and_time
        self.jumping_mean_ivals = jumping_mean_ivals
        self.select_ival = select_ival
        self.averaging_samples = averaging_samples
        self.input_type = mne.BaseEpochs
        self.rescale_to_uv = rescale_to_uv
        self.scaling = 1e6 if self.rescale_to_uv else 1
        self.pool_times = pool_times
        self.to_numpy_only = to_numpy_only
        self.copy = copy
        self.mne_scaler = mne_scaler
        if self.select_ival is None and self.jumping_mean_ivals is None:
            raise ValueError("jumping_mean_ivals or select_ival is required")

    def fit(self, X, y=None):
        """fit."""
        return self

    def transform(self, X: mne.BaseEpochs):
        """transform."""
        e = X.copy() if self.copy else X
        if self.to_numpy_only:
            X = e.get_data() * self.scaling
            return X
        if self.jumping_mean_ivals is not None:
            self.averaging_samples = np.zeros(len(self.jumping_mean_ivals))
            X = e.get_data() * self.scaling
            new_X = np.zeros((X.shape[0], X.shape[1], len(self.jumping_mean_ivals)))
            for i, ival in enumerate(self.jumping_mean_ivals):
                np_idx = e.time_as_index(ival)
                idx = list(range(np_idx[0], np_idx[1]))
                self.averaging_samples[i] = len(idx)
                new_X[:, :, i] = np.mean(X[:, :, idx], axis=2)
            X = new_X
        elif self.select_ival is not None:
            e.crop(tmin=self.select_ival[0], tmax=self.select_ival[1])
            X = e.get_data() * self.scaling
        elif self.pool_times:
            X = e.get_data() * self.scaling
            raise ValueError("This should never be entered though.")
        else:
            raise ValueError("In the constructor, pass either select ival or jumping means.")
        if self.mne_scaler is not None:
            X = self.mne_scaler.fit_transform(X)
        if self.permute_channels_and_time and not self.pool_times:
            X = X.transpose((0, 2, 1))
        if self.pool_times:
            X = np.reshape(X, (-1, X.shape[1]))
        else:
            X = np.reshape(X, (X.shape[0], -1))
        return X
