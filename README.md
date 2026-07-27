# Machine-learning-guided process-relevant replacement of Pd catalysts for furfural hydrogenation

This repository contains the supplementary Python code, datasets, trained models, and analysis results used in the intepretable machine-learning study of furfural hydrogenation to tetrahydrofurfuryl alcohol (THFA).

The repository provides the complete computational workflow for baseline model evaluation, grid-based hyperparameter optimization, optimized-model evaluation, interpretation machine learning-based analysis, catalyst-candidate assessment, figure generation, bootstrap-based uncertainty analysis, and process-level techno-economic assessment.

## Repository structure

```text
.
├── base_model/
│   ├── Model training and evaluation scripts
│   └── output/
│       └── Saved trained baseline models
│
├── hyperparameter_tuning/
│   ├── Grid-based hyperparameter optimization scripts
│   └── output/
│       └── Saved models trainined using the optimized hyperparameters
│
├── bootstrap_output/
│   └── Intermediate results required to generate Supplementary Information Figure 8
│
├── dataset/
│   ├── Preprocessed literature-derived dataset
│   ├── Training and test datasets for the input features and target variable
│   └── Pickle file containing the atomic weights of the active metals
│
├── result_figures_NCE.py
├── utils.py
├── environment.yml
└── README.md
```

The base_model/output directory contains the saved results of the baseline machine-learning models.

The hyperparameter_tuning/output directory contains the optimized hyperparameter configurations and the saved models trained using the selected hyperparameters.

## Python environment

The analyses were conducted using Python 3.10.18.

The required Conda environment can be created using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate supp_code
```

The principal Python packages used in this repository include:

* NumPy
* pandas
* SciPy
* scikit-learn
* XGBoost
* CatBoost
* LightGBM
* SHAP
* Matplotlib
* seaborn
* joblib

## Analysis workflow

### 1. Baseline model evaluation

The scripts in the `base_model` directory train and evaluate the baseline machine-learning models considered in this study.

All models are evaluated using the same training and test datasets to provide a consistent comparison of their predictive performance. The resulting predictions, evaluation metrics, and model checkpoints are stored in the corresponding `output` directory.

### 2. Grid-based hyperparameter optimization

The scripts in the `hyperparameter_tuning` directory conduct grid-based hyperparameter optimization for the machine-learning algorithms considered in this study.

The purpose of this step is to identify the hyperparameter configuration that provides the best cross-validation performance for each model. The model corresponding to the selected hyperparameters is then evaluated using the fixed test dataset.

The optimized hyperparameter configurations and saved models are stored in the corresponding `output` directory.

### 3. Figure generation and model analysis

The `result_figures_NCE.py` script contains the model-analysis and visualization procedures used to generate the computational figures reported in the manuscript and Supplementary Information.

The script reproduces all figures presented in the manuscript and Supplementary Information except for integrated workflow, experimental characterization figures, and the process-flow diagram.

The analyses and visualizations included in this script cover:

* predictive-performance evaluation of the optimized XGBoost model (Fig. 2a)
* SHAP-based feature interpretation (Fig. 2b)
* partial-dependence analysis of catalyst-preparation and operating variables (Fig. 2c–g)
* catalyst-family partial dependence responses to operating variables (Fig. 3a–c)
* Pd-relative operating-window mapping and catalyst-candidate evaluation based on positive-region fraction and average yield gap (Fig. 3d–f)
* comparison of model-predicted and experimental catalyst performance (Fig. 4a)
* catalyst yield–cost and minimum-selling-price comparisons (Fig. 6b,c)
* dataset-distribution analysis (Supplementary Fig. 1)
* parity-plot comparison of the nine machine-learning models (Supplementary Fig. 2)
* SHAP-analysis based on stratified operating time (Supplementary Figs. 3)
* Mean absolute SHAP value and category-level SHAP analysis (Supplementary Figs. 4)
* covariate-overlap assessment and overlap-weighted catalyst-family comparison (Supplementary Figs. 5 and 6)
* Pd-relative operating-window mapping of all catalyst candidates (Supplementary Fig. 7)
* bootstrap-based stability analysis of candidate prioritization (Supplementary Fig. 8)
* process-level cost breakdown and economic sensitivity analyses (Supplementary Figs. 17–19)

The script uses the datasets, saved models, optimized hyperparameters, and intermediate analysis outputs provided in the corresponding repository directories.

### 4. Bootstrap uncertainty analysis

The `bootstrap_output` directory contains the intermediate results required to reproduce Supplementary Information Figure 8.

These results were obtained through repeated bootstrap resampling and model refitting. They are used to evaluate the uncertainty and stability of the Pd-relative operating-window analysis and catalyst-candidate assessment.

The associated post-processing and visualization procedures are included in `result_figures_NCE.py`.

## Dataset

The `dataset` directory contains the data files required to reproduce the machine-learning analyses.

The provided files include:

* the preprocessed literature-derived dataset
* the training dataset for the input features
* the test dataset for the input features
* the training dataset for the target variable
* the test dataset for the target variable
* a pickle file containing the atomic weights of the active metals

The provided training and test datasets correspond to the fixed data split used for model development and evaluation in this study.

## Reproducibility

A fixed random seed of 23 was used where applicable for data splitting, model training, hyperparameter optimization, and model evaluation.

The `environment.yml` file specifies the principal package versions used in the analysis.

Minor numerical differences may occur depending on the operating system, processor architecture, parallel-computing settings, and package builds.
