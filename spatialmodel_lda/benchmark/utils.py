# Created by chris at 28.03.23 18:29
from functools import partial

from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from spatialmodel_lda.classification.feature_preprocessing import EpochsVectorizer
from spatialmodel_lda.classification.shrinkage_lda import (
    ShrinkageLinearDiscriminantAnalysis as ShrinkageLDA,
)


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
        "slda": partial(ShrinkageLDA, calc_oracle_cov=False, calc_oracle_mean=calc_oracle_mean),
    }
    scm_gammas = [None]
    for key, clf in clf_presets.items():
        for scm_gamma in scm_gammas:
            classifiers[f"{key}__sgamma({scm_gamma})_omean({calc_oracle_mean})"] = clf(
                scm_gamma=scm_gamma
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
