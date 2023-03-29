# Created by chris at 24.03.23 21:19
import argparse
import os
import time
import uuid
import warnings
from datetime import datetime as dt
from pathlib import Path
from shutil import copyfile

import moabb
import numpy as np
import yaml
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300

from spatialmodel_lda.benchmark.p300 import RejectP300
from spatialmodel_lda.benchmark.utils import create_erp_bench_debug

LOCAL_CONFIG_FILE = "local_config.yaml"
ANALYSIS_CONFIG_FILE = "analysis_config.yaml"

t0 = time.time()

with open(LOCAL_CONFIG_FILE, "r") as conf_f:
    local_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

RESULTS_ROOT = Path(local_cfg["results_root"])
RESULTS_RUN_NAME = dt.strftime(dt.now(), "%Y-%m-%d")
RESULTS_GROUPING = f"{uuid.uuid4()}"
RESULTS_FOLDER = (
    RESULTS_ROOT / local_cfg["benchmark_meta_name"] / RESULTS_RUN_NAME / RESULTS_GROUPING
)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
copyfile(ANALYSIS_CONFIG_FILE, RESULTS_FOLDER / ANALYSIS_CONFIG_FILE)
with open(RESULTS_FOLDER / ANALYSIS_CONFIG_FILE, "r") as conf_f:  # use local analysis config
    ana_cfg = yaml.load(conf_f, Loader=yaml.FullLoader)

VALID_DATASETS = [
    "bnci_1",
    "spot_pooled",
    "llp",
    "mix",
]
from moabb.datasets import BNCI2014009 as bnci_1

# TODO use all in Moabb available Datasets

bci_paradigm = "erp"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("dataset", help=f"Name of the dataset. Valid names: {VALID_DATASETS}")
parser.add_argument(
    "subjects_sessions",
    help="[Optional] Indices of subjects to benchmark.",
    type=str,
    nargs="*",
)

args = parser.parse_args()

dataset_name = args.dataset
if dataset_name not in VALID_DATASETS:
    raise ValueError(f"Invalid dataset name: {dataset_name}. Try one from {VALID_DATASETS}.")
subject_session_args = args.subjects_sessions
if len(subject_session_args) == 0:
    subjects = None
    sessions = None
# check whether args have format [subject, subject, ...] or [subject:session, subject:session, ...]
else:
    if np.all([":" in s for s in subject_session_args]):
        subjects = [int(s.split(":")[0]) for s in subject_session_args]
        sessions = [int(s.split(":")[1]) for s in subject_session_args]
    elif not np.any([":" in s for s in subject_session_args]):
        subjects = [int(s.split(":")[0]) for s in subject_session_args]
        sessions = None
    else:
        raise ValueError(
            "Currently, mixed subject:session and only subject syntax is not supported."
        )
print(f"Dataset: {dataset_name}")
print(f"Subject indices: {subjects}")
print(f"Sessions: {sessions}")

start_timestamp_as_str = dt.now().replace(microsecond=0).isoformat()

warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

moabb.set_log_level("warning")  # if on_nemo else moabb.set_log_level("info")

np.random.seed(43)


prepro_cfg = ana_cfg[bci_paradigm]["data_preprocessing"]
# TODO use get_benchmark_config from utils
bench_cfg = dict()
bench_cfg["paradigm"] = RejectP300(
    resample=prepro_cfg["sampling_rate"],
    fmin=prepro_cfg["fmin"],
    fmax=prepro_cfg["fmax"],
    reject_uv=prepro_cfg["reject_uv"],
    baseline=prepro_cfg["baseline"],
    reject_tmin=prepro_cfg["reject_tmin"],
    reject_tmax=prepro_cfg["reject_tmax"],
)
load_ival = [0, 1]


# if dataset_name == "spot_pooled":
#     d = SpotPilotData(load_single_trials=False)
#     d.interval = load_ival
#     if subjects is not None:
#         d.subject_list = [d.subject_list[i] for i in subjects]
#     n_channels = d.n_channels
if dataset_name == "bnci_1":
    d = bnci_1()
    d.interval = load_ival
    if subjects is not None:
        d.subject_list = [d.subject_list[i] for i in subjects]
    n_channels = 16
    d.n_channels = n_channels

bench_cfg["dataset"] = d
bench_cfg["n_channels"] = n_channels

if bci_paradigm == "erp":
    if hasattr(bench_cfg["dataset"], "stimulus_modality"):
        feature_preprocessing_key = bench_cfg["dataset"].stimulus_modality
    else:
        feature_preprocessing_key = ana_cfg[bci_paradigm]["fallback_modality"]
elif bci_paradigm == "imagery":
    raise NotImplementedError("Alternative feature preprocessing for Imagery not implemented yet")
    # TODO


pipelines = dict()
pipe_func = create_erp_bench_debug
pipelines.update(
    pipe_func(
        ana_cfg[bci_paradigm],
        feature_preprocessing_key,
        bench_cfg["n_channels"],
        calc_oracle_mean=False,
        model_corr_source=["oracle"],
    )
)

if subjects is not None:
    if len(subjects) > 5 and np.all(np.diff(subjects) == 1):
        subj_ids = f"[{np.min(subjects)}-{np.max(subjects)}]"
    else:
        subj_ids = subjects
else:
    subj_ids = "all"


identifier = (
    f"{dataset_name}_subj_{subj_ids}"
    f'_sess_{sessions if sessions is not None else "all"}_'
    f"{start_timestamp_as_str}".replace(" ", "")
)
unique_suffix = f"{identifier}_{uuid.uuid4()}"
debug_path = RESULTS_FOLDER / f"{identifier}_DEBUG.txt"

with open(debug_path, "w") as debug_f:
    debug_f.writelines([f"{l}: {os.environ[l]}\n" for l in sorted(os.environ)])

log_path = RESULTS_FOLDER / f"{identifier}_LOG.txt"

pipe_reprs = [name + ": " + pipe.steps[1][1].__repr__() + "\n" for name, pipe in pipelines.items()]
pipe_reprs = [f"Pipe constructor: {pipe_func.__name__}\n", *pipe_reprs]

overwrite = True
error_score = "raise"  # np.nan if on_nemo else "raise"

moabb.set_log_level("debug")

print(f"Benchmark starting at: {dt.now()}")
evaluation = WithinSessionEvaluation(
    paradigm=bench_cfg["paradigm"],
    datasets=bench_cfg["dataset"],
    suffix=unique_suffix,
    overwrite=overwrite,
    random_state=8,
    error_score=error_score,
    return_epochs=True,
)
results = evaluation.process(pipelines)
result_path = RESULTS_FOLDER / f"{identifier}_results.csv"

results.to_csv(result_path, encoding="utf-8", index=False)
t1 = time.time()
print(f"Benchmark run completed. Elapsed time: {(t1-t0)/3600} hours.")
