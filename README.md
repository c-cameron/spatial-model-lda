# spatial-model-lda
Code from my master thesis for an LDA classifier using spatial models of the EEG covariance matrix as a shrinkage target. The classifier conforms to sklearn and can be used as a replacement for other LDA classifiers. This implementation is intended for research use investigating hypothetical settings relating covariance estimation with the possibility to assess the conditioning and the accuracy of the estimated covariance matrices.

Included are also a simple example script and functions for benchmarking using [MOABB](https://github.com/NeuroTechX/moabb) with a custom P300 paradigm and Run-based evaluation procedure geared towards our use case in a BCI context.

TODO: Include scripts to fully reproduce the benchmarks in my thesis.  

## Installation [Ubuntu]

### 0. Clone this repository to your local machine.

### 1. Set up the virtual environment.
 a) If using poetry you can set up the environment and install all dependencies using `Poetry install`
 
 b)  Otherwise just 
  1. Create virtual environment: `python3 -m venv spatial_model_lda_venv`
  2. Activate virtual environment: `. spatial_model_lda_venv/bin/activate`
  3. Install dependencies: 
  ```
  pip install --upgrade pip==22.3.1 
  pip install -r requirements.txt
   ```
### 2. Set up the configuration files.
 Copy `scripts/local_config.yaml.example` to `scripts/local_config.yaml` and enter the `results_root` path where you want to store the benchmark results on your local machine. T

The `analysis_config.yaml` file defines the EEG preprocessing and the time intervals used for feature extraction. Setting the parameters for spatial modeling is not implemented in this version.


