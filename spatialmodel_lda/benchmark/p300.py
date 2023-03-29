# Created by chris at 29.03.23 18:57
import logging
import os
import pickle
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from moabb.paradigms.p300 import P300

log = logging.getLogger(__name__)


class RejectP300(P300):
    """# Extending MOABBS P300 class to include rejection methods and return additional data"""

    def __init__(
        self, reject_tmin=None, reject_tmax=None, reject_uv=None, reject_from_eog=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.reject_uv = reject_uv
        self.reject_from_eog = reject_from_eog
        if reject_tmin is not None:
            if reject_tmin <= self.tmin:
                raise ValueError(f"reject_tmin must be greater or equal to tmin:{self.tmin}")
        if reject_tmax is not None:
            if reject_tmax <= self.tmin:
                raise ValueError(f"reject_tmax must be greater than tmin: {self.tmin}")
        if reject_tmin is not None and reject_tmax is not None:
            if reject_tmin >= reject_tmax:
                raise ValueError("reject_tmax must be greater than reject_tmin")
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax

    def process_raw(self, raw, dataset, return_epochs=False, return_runs=False):
        # find the events, first check stim_channels then annotations
        if self.reject_from_eog:
            if isinstance(self.reject_from_eog, dict):
                # If its a dict, assuming its kwargs, no further checking.
                eog_kwargs = self.reject_from_eog
            elif isinstance(self.reject_from_eog, str):
                eog_kwargs = {"ch_name": self.reject_from_eog}
                if eog_kwargs["ch_name"] not in raw.ch_names:
                    raise ValueError(f"{eog_kwargs['ch_name']} not in channels ")
            elif self.reject_from_eog == True:
                eog_kwargs = {"ch_name": None}
            # EOG_CHANNEL = "EOGvu"

            time_pre_blink = 0.25
            blink_length = 0.7
            eog_events = mne.preprocessing.find_eog_events(raw, **eog_kwargs)
            onsets = eog_events[:, 0] / raw.info["sfreq"] - time_pre_blink
            durations = [blink_length] * len(eog_events)
            descriptions = ["bad blink"] * len(eog_events)
            blink_annot = mne.Annotations(
                onsets, durations, descriptions, orig_time=raw.info["meas_date"]
            )
            raw.set_annotations(raw.annotations + blink_annot)

        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(raw, verbose=False)

        # picks channels
        channels = () if self.channels is None else self.channels
        picks = mne.pick_types(raw.info, eeg=True, stim=False, include=channels)
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(raw.info["ch_names"], include=channels, ordered=True)

        # get event id
        event_id = self.used_events(dataset)

        # pick events, based on event_id
        try:
            if type(event_id["Target"]) is list and type(event_id["NonTarget"]) == list:
                event_id_new = dict(Target=1, NonTarget=0)
                events = mne.merge_events(events, event_id["Target"], 1)
                events = mne.merge_events(events, event_id["NonTarget"], 0)
                event_id = event_id_new
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]
        if self.reject_tmax is not None:
            if self.reject_tmax >= dataset.interval[1]:
                raise ValueError("reject_tmax needs to be shorter than tmax")
        X = []
        runs = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            raw_f = raw.copy().filter(fmin, fmax, method="iir", picks=picks, verbose=False)
            # epoch data
            baseline = self.baseline
            if baseline is not None:
                baseline = (
                    self.baseline[0] + dataset.interval[0],
                    self.baseline[1] + dataset.interval[0],
                )
                bmin = baseline[0] if baseline[0] < tmin else tmin
                bmax = baseline[1] if baseline[1] > tmax else tmax
            else:
                bmin = tmin
                bmax = tmax
            epoching_kwargs = dict(
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                reject_tmin=self.reject_tmin,
                reject_tmax=self.reject_tmax,
                proj=False,
                baseline=baseline,
                preload=True,
                verbose=False,
                picks=picks,
                on_missing="ignore",
                reject_by_annotation=self.reject_from_eog,
            )
            epochs = mne.Epochs(
                raw_f,
                events,
                **epoching_kwargs,
            )
            if self.reject_uv is not None:
                epochs.drop_bad(dict(eeg=self.reject_uv / 1e6))

            if len(epochs) == 0:
                print(f"All epochs were removed in run {run}. Are you sure this is right?")
            if bmin < tmin or bmax > tmax:
                epochs.crop(tmin=tmin, tmax=tmax)
            if self.resample is not None:
                epochs = epochs.resample(self.resample)
            # rescale to work with uV
            runs.append(raw_f if return_runs else None)
            if return_epochs:
                X.append(epochs)
            else:
                X.append(dataset.unit_factor * epochs.get_data())

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        # if only one band, return a 3D array, otherwise return a 4D
        if return_epochs:
            X = mne.concatenate_epochs(X)
        elif len(self.filters) == 1:
            X = X[0]
        else:
            X = np.array(X).transpose((1, 2, 3, 0))

        metadata = pd.DataFrame(index=range(len(labels)))

        return X, labels, metadata, (runs, events, epoching_kwargs)

    def get_data(self, dataset, subjects=None, return_epochs=False, return_runs=False, cache=False):
        """
        Return the data for a list of subject.

        return the data, labels and a dataframe with metadata. the dataframe
        will contain at least the following columns

        - subject : the subject indice
        - session : the session indice
        - run : the run indice

        parameters
        ----------
        dataset:
            A dataset instance.
        subjects: List of int
            List of subject number
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        return_runs: boolean
            If True, the processed runs before epoching are also returned
        cache: boolean
            If True, paradigm processed data is stored in /tmp and read from
            there if available. WARNING: does not notice changes in preprocessing
            and could lead to disk space issues.

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
                  if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata.
        """

        if cache:
            tmp = Path("/tmp/moabb/cache")
            os.makedirs(tmp, exist_ok=True)
            prefix = f"{dataset.__class__.__name__}_{subjects}"
            try:
                with open(tmp / f"{prefix}.pkl", "rb") as pklf:
                    d = pickle.load(pklf)
                labels = d["labels"]
                metadata = d["metadata"]
                X = d["X"]
                raws = None
                log.warning("Using cached data: Beware that it might not be the data you want!")
                return X, labels, metadata, raws
            except Exception as e:
                print("Could not read cached data. Preprocessing from scratch.")
                print(e)

        if not self.is_valid(dataset):
            message = "Dataset {} is not valid for paradigm".format(dataset.code)
            raise AssertionError(message)

        data = dataset.get_data(subjects)
        self.prepare_process(dataset)

        X = [] if return_epochs else np.array([])
        labels = []
        metadata = []
        processed_runs = []
        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    proc = self.process_raw(
                        raw, dataset, return_epochs=return_epochs, return_runs=return_runs
                    )

                    if proc is None:
                        # this mean the run did not contain any selected event
                        # go to next
                        continue

                    x, lbs, met, praw = proc
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run

                    # grow X and labels in a memory efficient way. can be slow
                    if len(x) > 0:
                        metadata.append(met)
                        labels = np.append(labels, lbs, axis=0)
                        if return_epochs:
                            X.append(x)
                        else:
                            X = np.append(X, x, axis=0)
                        if return_runs:
                            processed_runs.append(praw)
                    else:
                        log.warning(
                            f"All epochs were removed in run {run}. Are you sure the rejection settings are right?"
                        )

        metadata = pd.concat(metadata, ignore_index=True)
        if return_epochs:
            # TODO: how do we handle filter-bank for ERP? Should we at all?
            if type(X[0]) is list:
                X = [x[0] for x in X]
            X = mne.concatenate_epochs(X)

        if cache:
            tmp = Path("/tmp/moabb/cache")
            os.makedirs(tmp, exist_ok=True)
            prefix = f"{dataset.__class__.__name__}_{subjects}"
            try:
                # epochs.save(tmp / f"{prefix}-epo.fif", overwrite=True)
                with open(tmp / f"{prefix}.pkl", "wb") as pklf:
                    pickle.dump(
                        {
                            "labels": labels,
                            "metadata": metadata,
                            "X": X,
                        },
                        pklf,
                    )
            except Exception as e:
                print("Could not store cached data")
                print(e)
        if return_runs:
            return X, labels, metadata, processed_runs
        else:
            return X, labels, metadata
