# Created by chris at 28.03.23 18:29
import itertools
from functools import partial
from itertools import product

from moabb.datasets import BNCI2014008 as bnci_als
from moabb.datasets import BNCI2014009 as bnci_1
from moabb.datasets import BNCI2015003 as bnci_2
from moabb.datasets import (
    EPFLP300,
    Huebner2017,
    Huebner2018,
    Lee2019_ERP,
    Sosulski2019,
    bi2013a,
)
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline

import spatialmodel_lda.classification.fit_functions as ff
from spatialmodel_lda.benchmark.p300 import RejectP300
from spatialmodel_lda.classification.feature_preprocessing import EpochsVectorizer
from spatialmodel_lda.classification.shrinkage_lda import (
    ShrinkageLinearDiscriminantAnalysis as ShrinkageLDA,
)
from spatialmodel_lda.classification.spatial_model_lda import SpatialOnlyLda

allowed_datasets = dict()
allowed_datasets["erp"] = [
    "spot",
    "llp",
    "mix",
    "epfl",
    "bnci_1",
    "bnci_als",
    "bnci_2",
    "braininvaders",
    "lee",
]
# allowed_datasets["imagery"] = ["physionet_mi", "weibo2014", "munich_mi", "schirrmeister2017"]


def get_allowed_datasets():
    return allowed_datasets


def get_erp_benchmark_config(dataset_name, cfg_prepro, subjects=None, sessions=None):
    benchmark_cfg = dict()
    paradigms = dict()
    # check if any preprocessing config is a list, e.g. have multiple values
    if any([type(v) is list for v in cfg_prepro.values()]):
        # get keys that have multiple values
        multi_keys = [(k, v)[0] for k, v in cfg_prepro.items() if type(v) is list]
        # generate key/value pairs for naming the paradigm
        paradigm_keys = itertools.product(*[[key] * len(cfg_prepro[key]) for key in multi_keys])
        paradigm_vals = itertools.product(*[cfg_prepro[key] for key in multi_keys])
        for pkey, pval in zip(paradigm_keys, paradigm_vals):
            # combine all multi-value keys with the specific value
            paradigm_name = "__".join([f"{k}_{v}" for k, v in zip(pkey, pval)])
            cfg_prepro.update(dict(zip(pkey, pval)))
            paradigms[paradigm_name] = RejectP300(
                resample=cfg_prepro["sampling_rate"],
                fmin=cfg_prepro["fmin"],
                fmax=cfg_prepro["fmax"],
                reject_uv=cfg_prepro["reject_uv"],
                baseline=cfg_prepro["baseline"],
                reject_tmin=cfg_prepro["reject_tmin"],
                reject_tmax=cfg_prepro["reject_tmax"],
            )
    else:
        paradigms["default"] = RejectP300(
            resample=cfg_prepro["sampling_rate"],
            fmin=cfg_prepro["fmin"],
            fmax=cfg_prepro["fmax"],
            reject_uv=cfg_prepro["reject_uv"],
            baseline=cfg_prepro["baseline"],
            reject_tmin=cfg_prepro["reject_tmin"],
            reject_tmax=cfg_prepro["reject_tmax"],
        )
    if dataset_name == "spot":
        d = Sosulski2019()
    elif dataset_name == "llp":
        d = Huebner2017()
    elif dataset_name == "mix":
        d = Huebner2018()
    elif dataset_name == "epfl":
        d = EPFLP300()
        d.unit_factor = 1
        d.n_channels = 32
    elif dataset_name == "bnci_1":
        d = bnci_1()
        d.n_channels = 16
    elif dataset_name == "bnci_als":
        d = bnci_als()
        d.n_channels = 8
    elif dataset_name == "bnci_2":
        d = bnci_2()
        d.n_channels = 8
    elif dataset_name == "braininvaders":
        d = bi2013a(NonAdaptive=True, Adaptive=True, Training=True, Online=True)
        d.n_channels = 16
    elif dataset_name == "lee":
        d = Lee2019_ERP(sessions=sessions)
        d.n_channels = 62
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")

    load_ival = [0, 1]
    d.interval = load_ival
    if subjects is not None:
        d.subject_list = [d.subject_list[i] for i in subjects]
    benchmark_cfg["dataset"] = d
    benchmark_cfg["n_channels"] = d.n_channels
    benchmark_cfg["paradigms"] = paradigms
    return benchmark_cfg


def get_info_params():
    matrices = [
        "sample_cov",
        "model_cov",
        "avg_cov",
    ]
    matrix_params = ["cond", "nmse_dist", "logeuc_dist"]
    params = [
        "scm_gamma",
        "model_gamma",
        *["_".join(combi) for combi in product(matrices, matrix_params)],
    ]
    stats = ["mean", "std", "min", "median", "max"]
    info_columns = [
        "model_cov_cond",
        "model_corr_cond",
        "model_cond_gamma",
        *["_".join(combi) for combi in product(params, stats)],
    ]
    return info_columns


def create_erp_bench_debug(
    cfg,
    feature_preprocessing_key,
    n_channels,
    calc_oracle_mean=False,
):
    cfg_vect = cfg[feature_preprocessing_key]["feature_preprocessing"]
    c_sel = cfg_vect["select_ival"]

    vectorizers = dict()
    for key in c_sel:
        vectorizers[f"{key}"] = dict(
            vec=EpochsVectorizer(select_ival=c_sel[key]["ival"]),
            D=c_sel[key]["D"],
        )
    classifiers = dict()
    clf_presets = {
        "spatialda": partial(
            SpatialOnlyLda,
            distance_metric="3d",
            calc_oracle_cov=True,
            calc_oracle_mean=calc_oracle_mean,
        ),
        "slda": partial(ShrinkageLDA, calc_oracle_cov=False, calc_oracle_mean=calc_oracle_mean),
    }
    scm_gammas = [None]
    for key, clf in clf_presets.items():
        for scm_gamma in scm_gammas:
            classifiers[f"{key}__sgamma({scm_gamma})_omean({calc_oracle_mean})"] = clf(
                scm_gamma=scm_gamma
            )
    fit_functions = {"poly2-c1": ff.poly2_func_c1}
    spatial_cfg = dict(
        scm_gamma=[0],
        model_gamma=[0.05, 1],
        fit=list(fit_functions.keys()),
        model_var_source=["scm"],
        model_corr_source=["decoupled"],
    )
    param_grid = ParameterGrid(spatial_cfg)

    c_key = "spatialda"
    for p in param_grid:
        new_c_key = f"{c_key}__corrsource({p['model_corr_source']})_mgamma({p['model_gamma']})_fit({p['fit']})_omean({calc_oracle_mean})"
        classifiers[new_c_key] = clf_presets[c_key](
            scm_gamma=p["scm_gamma"],
            model_gamma=p["model_gamma"],
            spatial_fit_function=fit_functions[p["fit"]],
            model_var_source=p["model_var_source"],
            model_corr_source=p["model_corr_source"],
        )
    pipelines = dict()
    for v_key in vectorizers.keys():
        D = vectorizers[v_key]["D"]
        vec = vectorizers[v_key]["vec"]
        for c_key in classifiers.keys():
            clf = clone(classifiers[c_key])
            clf.n_times = D
            clf.n_channels = n_channels
            clf.calc_oracle_mean = calc_oracle_mean
            pipelines[c_key] = make_pipeline(vec, clf)
    return pipelines
