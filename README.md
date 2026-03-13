# README

## Introduction

* This repository contains code and analyses to replicate findings from 'Training Employees to Love Their Job? A Forest-Based Analysis of Job Training and Underemployment in the UK' (Gonzalez Galvan, 2026). The dataset used is the October 2024-September 2025 Annual Population Survey.

## Environment setup and package installation

* In your terminal, from the project root:
1. `conda env create -f environment.yml` to create the conda environment.
2. `conda activate gg570_d200` to activate the environment. Select the corresponding kernel within your IDE.
3. `pip install -e .` to install the gg570_d200 package in editable mode.

## After setup and installation

### Data
* Data is made available in compressed form within the `data/` folder.
* The user may decompress the file themselves if they choose to do so, ensuring that the decompressed `apsp_o24s25_eul_pwta22.dta` file lies within the `data/` folder, although, ideally, the user should decompress the file by following the code in `analyses/exploratory_analysis.ipynb`.
* If, for any reason, data decompression issues are encountered, the user can download the data [here](https://datacatalogue.ukdataservice.ac.uk/datasets/dataset/85aba2e9-5113-e840-f2aa-c6e16b2b7c6a).
    * The user must have a UK Data Service account to do so.

### Execution logic
* The following notebooks (in `analyses/`) can be used independently (once `data/df.csv` and `data/df_scaled.csv` exist. They are created by `analyses/exploratory_analysis.ipynb`, but they already exist in the containterised version of the project). If logical sequential execution is desired, the user should follow the following order:
1. `exploratory_analysis.ipynb` decompresses the data, reads it, and process it by wrangling features and discarding observations, where relevant. A standard-scaled version of the data is also produced and overlap metrics are computed.
2. `synthetic_data.ipynb` performs the Monte Carlo simulation and generates relevant visuals.
3. `ate_estimation.ipynb` performs ForestRiesz and CausalForestDML-based ATE and GATE estimations.

* The Jupyter notebooks rely on `analyses/` being the current working directory.
* Helper modules (in `gg570_d200/auxiliary_functions/`) are called from within these files. Additionally, `gg570_d200/external_code/forestriesz.py` is also refered to—this file contains code that does NOT belong to me, and an adequate reference and usage license is provided in the file.
* Outputs are generally saved to the `results/` folder. However, the current containerised version of this project _already contains_ all relevant outputs, which the user may choose to inspect without running any code.