# Created by chris at 28.03.23 18:29
from functools import partial
from itertools import product

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline

import spatialmodel_lda.classification.fit_functions as ff
from spatialmodel_lda.classification.feature_preprocessing import EpochsVectorizer
from spatialmodel_lda.classification.shrinkage_lda import (
    ShrinkageLinearDiscriminantAnalysis as ShrinkageLDA,
)
from spatialmodel_lda.classification.spatial_model_lda import SpatialOnlyLda


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
