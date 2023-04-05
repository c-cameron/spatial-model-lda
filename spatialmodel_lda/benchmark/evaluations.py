import logging
from copy import deepcopy
from time import time

import mne
import numpy as np
from moabb.evaluations.base import BaseEvaluation
from moabb.paradigms import BaseP300
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

# from spot.utils.classification import balanced_class_accuracy


def classwise_accuracies(y, y_pred, tar_idx, nontar_idx):
    tar_acc = np.sum(y_pred[tar_idx] == 1) / len(tar_idx)
    nontar_acc = np.sum(y_pred[nontar_idx] == 0) / len(nontar_idx)
    return tar_acc, nontar_acc


def balanced_class_accuracy(y, cout):
    y_sort = cout.copy()
    y_sort.sort()
    tar_acc = np.zeros((y.shape[0] + 1,))
    nontar_acc = np.zeros((y.shape[0] + 1,))
    tar_idx = np.where(y == 1)[0]
    nontar_idx = np.where(y == 0)[0]
    threshs = [*y_sort, np.inf]

    for i, thr in enumerate(threshs):
        clf_y = np.int64(cout >= thr)
        tar, nontar = classwise_accuracies(y, clf_y, tar_idx, nontar_idx)
        tar_acc[i] = tar
        nontar_acc[i] = nontar
    bal_acc = (tar_acc + nontar_acc) / 2
    return bal_acc, tar_acc, nontar_acc, threshs


log = logging.getLogger(__name__)


class RunbasedBciEvaluation(BaseEvaluation):
    """Evaluation for evaluating classifiers in an online setting

    returns Score computed within each recording session with first x-% of data

    """

    def __init__(
        self,
        train_ratios=None,
        n_perms=1,
        group_runs=False,
        load_aux=False,
        pass_oracle_data=False,
        pass_resting_data=False,
        limit_train_epos=None,
        limit_oracle_epos=None,
        oracle_reject_uv=False,
        dispatch_plots=None,
        chronological=False,
        time_slice=None,
        info_columns=None,
        **kwargs,
    ):
        if train_ratios is None:
            train_ratios = [0.2]
        self.train_ratios = train_ratios
        self.n_perms = n_perms
        self.group_runs = group_runs
        self.load_aux = load_aux
        self.pass_oracle_data = pass_oracle_data
        self.pass_resting_data = pass_resting_data
        self.limit_train_epos = limit_train_epos if limit_train_epos is not None else [np.inf]
        self.limit_oracle_epos = limit_oracle_epos if limit_oracle_epos is not None else [np.inf]
        self.oracle_reject_uv = oracle_reject_uv
        self.dispatch_plots = dispatch_plots
        self.chronological = chronological
        if chronological:
            self.train_ratios = [1 - 1 / n_perms]
        self.time_slice = time_slice
        additional_columns = [
            "score_bal_acc",
            "score_max",
            "score_recalc_bias",
            "score_auc",
            "train_ratio",
            "n_correct",
            "permutation",
            "n_samples_test",
            "dropped_epochs",
            "samples_limit",
            "oracle_limit",
            "oracle_reject_uv",
        ]
        self.info_columns = info_columns
        if info_columns is not None:
            additional_columns.extend(self.info_columns)
        #
        super().__init__(additional_columns=additional_columns, **kwargs)

    def load_resting_data(self, dataset, subject):
        if dataset.code != "Spot Pilot P300 dataset pooled":
            raise NotImplementedError
        raw_f = dataset.get_resting_eeg_data(subject=subject, artifact_reject=True)
        annotation = mne.Annotations(0, 4, "bad blink")
        raw_f.set_annotations(raw_f.annotations + annotation)
        resting_X = raw_f.get_data(reject_by_annotation="omit") * 1e6
        resting_info = raw_f.info["ch_names"]
        return resting_X, resting_info

    def evaluate(self, dataset, pipelines):
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-Runbased"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)
            if len(run_pipes) == 0:
                continue

            # get the data
            X_all, y_all, metadata_all = self.paradigm.get_data(
                dataset,
                [subject],
                return_epochs=True,
            )

            if self.pass_resting_data:
                if hasattr(dataset, "get_resting_eeg_data"):
                    resting_X, resting_info = self.load_resting_data(dataset, subject)
                else:
                    raise NotImplementedError("Loading resting data is only implemented for Spot")
            n_sessions = len(np.unique(metadata_all.session))
            n_pipes = len(run_pipes)
            log.info(f"Subject: {subject}, {n_pipes=}, {n_sessions=}")
            n_total = len(X_all.drop_log)
            n_dropped = n_total - len(X_all)
            log.info(f"Dropped {n_dropped} out of {n_total} epochs")
            np.random.seed(self.random_state)
            classes = np.unique(y_all)
            if min(self.limit_train_epos) < len(classes):
                raise ValueError(
                    f"Cannot run Evaluation when smallest train epo subset cannot contain all classes"
                )
            for sess_i, session in enumerate(np.unique(metadata_all.session)):
                log.info(f" Session: {session}")
                ix = metadata_all.session == session
                metadata = metadata_all[ix]
                X = X_all[ix]
                y = y_all[ix]
                for p in range(self.n_perms):
                    log.info(f"  Permutation: {p}")
                    clf_aux_data = dict()
                    for tr in self.train_ratios:
                        log.info(f"   Train ratio: {tr}")
                        groups = metadata.loc[:, "run"]
                        group_idx = groups.unique()
                        if len(group_idx) == 1 and not self.chronological:
                            # raise ValueError("No run information available.")
                            log.info(f"Creating fake group! No run information available.")
                            groups.iloc[len(groups) // 2 :] = "fake_group"
                            group_idx = groups.unique()
                            # TODO implement fallback for datasets that do not support this
                        min_train_size = 1 / len(group_idx)
                        if tr < min_train_size:
                            log.info(
                                f"Using {min_train_size} instead of {tr} as we only have {len(group_idx)} runs."
                            )
                            tr = min_train_size
                        if not self.chronological:
                            cvsplit = GroupShuffleSplit(n_splits=1, train_size=tr)
                            cv_iter = cvsplit.split(X, y, groups)
                            train_idx, test_idx = next(cv_iter)
                        else:
                            cvsplit = TimeSeriesSplit(n_splits=self.n_perms)
                            cv_iter = cvsplit.split(X, y, groups)
                            train_idx, test_idx = list(cv_iter)[p]

                        evaluated_ltes = []
                        for lte in self.limit_train_epos:
                            limitepochs = min(lte, len(train_idx))
                            if limitepochs in evaluated_ltes:
                                log.info(
                                    f"Skipping current limit epochs evaluation as it was already evaluated before"
                                )
                                continue
                            evaluated_ltes.append(limitepochs)
                            log.info(f"Using {limitepochs} of {len(train_idx)} training epochs.")
                            ti = train_idx[:limitepochs]
                            while len(np.unique(y[ti])) < len(classes):
                                log.info(f"Subset does not contain both classes. Offsetting.")
                                ti += 1

                            X_train = X[ti]
                            X_test = X[test_idx]
                            y_train = y[ti]
                            y_test = y[test_idx]

                            if self.pass_resting_data:
                                clf_aux_data["resting_data"] = dict(
                                    resting_data=resting_X, info=resting_info
                                )
                            if np.any(
                                [
                                    hasattr(clf, "use_spatial_info")
                                    for pipe in run_pipes.values()
                                    for clf in pipe
                                ]
                            ):
                                clf_aux_data["spatial_info"] = dict(
                                    montage=X.get_montage(), info=X.info
                                )
                            for limit_oracle_epos in self.limit_oracle_epos:
                                loe = min(
                                    limit_oracle_epos, len(train_idx)
                                )  # Sollte limit_oracle_epos + limitepochs Fur oracle data sein
                                if self.pass_oracle_data:
                                    # Index sollte in train_idx von limitepochs:loe sein
                                    log.info(
                                        f"Using {loe} of {len(train_idx)} training epochs for oracle data"
                                    )
                                    if self.oracle_reject_uv:
                                        inv_events = {k: v for v, k in X.event_id.items()}
                                        oracle_data = dict()
                                        oracle_data["x"] = (
                                            X[train_idx[:loe]]
                                            .copy()
                                            .drop_bad(dict(eeg=self.oracle_reject_uv / 1e6))
                                        )
                                        oracle_data["y"] = np.array(
                                            [inv_events[e] for e in oracle_data["x"].events[:, -1]]
                                        )
                                    else:
                                        oracle_data = dict(
                                            x=X[train_idx[:loe]], y=y[train_idx[:loe]]
                                        )

                                else:
                                    oracle_data = None
                                clf_aux_data["oracle_data"] = oracle_data

                                for name, clf in run_pipes.items():
                                    t_start = time()
                                    (
                                        score_bal_acc,
                                        score_max,
                                        score_recalc_bias,
                                        score_auc,
                                        n_correct,
                                    ), info_params = self.score(
                                        clf,
                                        X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        # oracle_data=oracle_data.copy(),
                                        clf_aux_data=clf_aux_data,
                                    )

                                    duration = time() - t_start
                                    res = {
                                        "time": duration,  # 5 fold CV
                                        "dataset": dataset,
                                        "subject": subject,
                                        "session": f"{session}",
                                        "score": score_auc,
                                        "score_bal_acc": score_bal_acc,
                                        "score_max": score_max,
                                        "score_recalc_bias": score_recalc_bias,
                                        "score_auc": score_auc,
                                        "permutation": p,
                                        "train_ratio": tr,
                                        "n_correct": n_correct,
                                        "n_samples": len(y_train),
                                        "n_samples_test": len(y_test),
                                        "n_channels": self.datasets[0].n_channels,
                                        "pipeline": name,
                                        # ADD EXTRA INFO HERE
                                        "dropped_epochs": n_dropped,
                                        "samples_limit": lte,
                                        "oracle_limit": loe,
                                        "oracle_reject_uv": self.oracle_reject_uv
                                        if self.oracle_reject_uv
                                        else np.nan,
                                    }

                                    for key, param in info_params.items():
                                        if not isinstance(
                                            param, (float, int, str, bool, np.number)
                                        ):
                                            raise ValueError(
                                                f"{key} is not a valid datatype. Only Int, Float, String or Booleans are allowed for the results"
                                            )
                                    res.update(info_params)
                                    yield res

    def score(self, clf, X_train, y_train, X_test, y_test, clf_aux_data=None):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        roc_auc = get_scorer("roc_auc")
        bal_acc = get_scorer("balanced_accuracy")

        try:
            if clf_aux_data:
                import inspect

                kwargs = dict()
                for name, step in clf.steps:
                    if hasattr(step, "use_spatial_info"):
                        for key, val in clf_aux_data["spatial_info"].items():
                            kwargs[f"{name}__{key}"] = val

                    if (
                        clf_aux_data["oracle_data"] is not None
                        and "oracle_data" in inspect.getfullargspec(step.fit).args
                    ):
                        if isinstance(self.paradigm, BaseP300):
                            oracle_dict = {}
                            oracle_dict["x"] = clf[:-1].fit_transform(
                                clf_aux_data["oracle_data"]["x"].copy()
                            )
                            oracle_dict["y"] = le.fit_transform(
                                clf_aux_data["oracle_data"]["y"].copy()
                            )
                        kwargs[f"{name}__oracle_data"] = oracle_dict

                    if hasattr(step, "use_resting_data") and self.pass_resting_data:
                        for key, val in clf_aux_data["resting_data"].items():
                            kwargs[f"{name}__{key}"] = val

                model = deepcopy(clf).fit(X_train, y_train, **kwargs)
            else:
                model = deepcopy(clf).fit(X_train, y_train)

            train_bias = model[-1].intercept_
            score_bal_acc = _score(model, X_test, y_test, bal_acc)
            score_auc = _score(model, X_test, y_test, roc_auc)

            # Calculate optimal bias based on test data
            cout = model.decision_function(X_test)
            bal_test, tar, ntar, thr_test = balanced_class_accuracy(y_test, cout)
            optimal_test_bias = thr_test[bal_test.argmax()]
            score_max = bal_test.max()

            # Recalculate optimal bias based on train data
            cout = model.decision_function(X_train)
            bal, tar, ntar, thr = balanced_class_accuracy(y_train, cout)
            optimal_train_bias = thr[bal.argmax()]
            # hotfix for array intercept with 1 element currently
            try:
                model[-1].intercept_[0] = optimal_train_bias
            except TypeError:
                model[-1].intercept_ = optimal_train_bias
            score_recalc_bias = _score(model, X_test, y_test, bal_acc)
            n_correct = np.min(np.unique(y_test, return_counts=True)[1]) * score_bal_acc

            # print(f"Diff between train/test bias:         {optimal_train_bias - train_bias}")
            # print(f"Diff between train/recalc_train bias: {optimal_test_bias - train_bias}")
            # print(f"Score with different Biases:\n  base:   {score}\n  recalc: {score_recalc}\n"
            #       f"  test:   {score_testcalc}")
            acc = score_bal_acc, score_max, score_recalc_bias, score_auc, n_correct

            info_params = dict()
            if self.info_columns:
                for step in model:
                    # Check if theres a quick hasattr+ True check
                    if hasattr(step, "return_info_params"):
                        if step.return_info_params:
                            info_params.update(step.info_params)
                for key in self.info_columns:
                    if key not in info_params.keys():
                        info_params[key] = np.nan

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if self.error_score == "raise":
                raise e
            elif self.error_score is np.nan:
                print(e)
                acc = np.nan, np.nan, np.nan, np.nan, np.nan
                info_params = {key: np.nan for key in self.info_columns}
        return acc, info_params

    def score_test(self, clf, X, y, scoring):
        le = LabelEncoder()
        y = le.fit_transform(y)

    def is_valid(self, dataset):
        return True
