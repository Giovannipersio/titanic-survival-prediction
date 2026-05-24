# Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-FFCC00?style=flat)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-3155A4?style=flat)](https://optuna.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/titanic)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)]()

## Overview

This repository contains a machine learning project for the Kaggle **Titanic - Machine Learning from Disaster** competition. The goal is to predict passenger survival from structured tabular data using exploratory data analysis, leakage-aware preprocessing, feature engineering, CatBoost modeling, Optuna hyperparameter optimization, local validation, out-of-fold diagnostics, and model explainability.

The project is designed as a portfolio-ready tabular classification workflow. It emphasizes reproducible experimentation, explicit handling of missing data, careful validation design, and cautious interpretation of model diagnostics.

## Project Objective

The objective is to build a supervised binary classification pipeline that predicts the `Survived` target:

- `0`: passenger did not survive
- `1`: passenger survived

The project demonstrates the following technical components:

- Exploratory Data Analysis (EDA)
- Missing-value treatment with statistics learned from fitting partitions
- Feature engineering from passenger names, family structure, age, fare, cabin, ticket, and class information
- CatBoost classification with native categorical handling
- Hyperparameter optimization with Optuna
- Stratified holdout validation
- Out-of-Fold (OOF) diagnostic evaluation
- CatBoost feature importance and SHAP-based model explainability
- Kaggle-compatible submission generation

## Dataset

The data comes from the Kaggle Titanic competition.

Raw Kaggle data files are not intended to be versioned in this repository. To reproduce the project, download the competition files from Kaggle and place them in the following paths:

```text
data/train.csv
data/test.csv
```

| File | Description |
| --- | --- |
| `train.csv` | Labeled training dataset containing the `Survived` target. |
| `test.csv` | Unlabeled Kaggle test dataset used to generate the submission file. |

## Repository Structure

```text
titanic-survival-prediction/
|-- data/                         # Local Kaggle data files
|-- notebooks/                    # Analysis, modeling, validation, and submission notebook
|   `-- titanic_catboost_optuna_pipeline.ipynb
|-- outputs/                      # Generated submission files
|-- requirements.txt              # Python dependencies
|-- make_env.py                   # Optional environment setup helper
|-- .gitignore
`-- README.md
```

## Methodology

### 1. Exploratory Data Analysis

The notebook inspects the target distribution, raw feature distributions, missing-value patterns, bivariate relationships with survival, and linear correlations. The EDA is descriptive and is used to motivate preprocessing and feature engineering decisions.

Key descriptive observations include:

- `Sex` and `Pclass` show strong observed associations with survival.
- `Age` and `Fare` show non-linear patterns, motivating grouped representations.
- `SibSp` and `Parch` capture related family-structure information.
- `Embarked` provides an additional categorical signal.
- High-cardinality fields such as `Ticket` may encode passenger grouping information, but require cautious interpretation.

### 2. Preprocessing

Missing values are handled with statistics learned from the relevant fitting partition. This is important for reducing leakage during local validation.

| Feature | Strategy |
| --- | --- |
| `Embarked` | Mode imputation. |
| `Fare` | Median imputation by `Pclass`, with global median fallback. |
| `Age` | Median imputation by `Sex` and `Pclass`, with global median fallback. |
| `Cabin` | Missing values replaced with explicit `U` category. |
| `PassengerId` | Removed from model features and preserved only for submission. |

For holdout validation, cross-validation, and OOF diagnostics, preprocessing is rebuilt from raw rows inside the relevant training partitions.

### 3. Feature Engineering

The modeling pipeline combines retained raw predictors with engineered features.

| Feature | Description |
| --- | --- |
| `Title` | Extracted from passenger names and grouped into common title categories. |
| `IsMarried` | Binary indicator derived from `Title == "Mrs"`. |
| `FamilySize` | Combined family-size measure: `SibSp + Parch + 1`. |
| `AgeBin` | Discretized age representation using fixed cut points. |
| `FareBin` | Discretized fare representation using fixed fare bands. |
| `Sex_Pclass` | Interaction between passenger sex and ticket class. |

The final model also retains selected raw variables such as `Name`, `Age`, `Ticket`, `Fare`, `Cabin`, `Embarked`, `Sex`, and `Pclass`. These features should be interpreted as part of the combined modeling pipeline rather than as individually proven contributors.

### 4. Modeling

The project uses `CatBoostClassifier` as the modeling family. CatBoost is well suited for this task because the feature matrix combines numerical predictors, categorical predictors, high-cardinality identifiers, and engineered categorical interactions.

Optuna is used to tune CatBoost hyperparameters with a TPE sampler. The search evaluates candidate configurations using stratified cross-validation within the internal training split.

### 5. Validation

The notebook separates validation signals into three categories:

| Evaluation | Purpose | Interpretation |
| --- | --- | --- |
| Holdout validation | Evaluates the selected model on a stratified 20% local validation split. | Local estimate for one split and one random seed. |
| Optuna cross-validation | Selects hyperparameters using stratified folds within the fitting split. | Model-selection criterion, not final leaderboard performance. |
| OOF diagnostic | Generates out-of-fold predictions across the labeled training set. | Post-selection diagnostic because hyperparameters are selected before the OOF loop. |
| Kaggle public score | Score from the public leaderboard after manual submission. | External public leaderboard signal, not a private leaderboard guarantee. |

## Reported Results

The following results are documented in the notebook and experiment log:

| Metric / Evaluation | Reported Value | Notes |
| --- | ---: | --- |
| Holdout accuracy | `0.7989` | Local stratified holdout validation. |
| OOF diagnostic accuracy | `~0.8474` | Post-selection OOF diagnostic reported in the notebook. |
| Kaggle public leaderboard score | `0.79904` | Best public score recorded for the current notebook experiment log. |

Local validation metrics, OOF diagnostics, and Kaggle public leaderboard scores answer different questions and should not be treated as interchangeable estimates.

## Model Explainability

The notebook includes two model-behavior diagnostics:

- CatBoost native feature importance
- SHAP values

These methods are used to inspect how the fitted model uses the available predictors. They do not establish causality and do not prove that any single feature independently improves generalization.

Features discussed in the explainability section include:

- `Title`
- `Sex_Pclass`
- `Ticket`
- `Age`
- `Fare`
- `Cabin`
- `FamilySize`
- `Pclass`

High-cardinality predictors such as `Ticket` and `Name` are interpreted cautiously. They may capture meaningful grouping structure, but they may also encode dataset-specific patterns.

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/Giovannipersio/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Kaggle data

Download the Titanic competition files from Kaggle and place them as:

```text
data/train.csv
data/test.csv
```

### 5. Run the notebook

Open and execute:

```text
notebooks/titanic_catboost_optuna_pipeline.ipynb
```

The notebook will run the EDA, preprocessing, feature engineering, model tuning, validation, explainability diagnostics, and submission generation.

### 6. Generate the submission

The Kaggle submission file is saved as:

```text
outputs/submission.csv
```

## Requirements

Main libraries used in this project:

| Library | Purpose |
| --- | --- |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Data splitting and evaluation metrics |
| `catboost` | Classification model |
| `optuna` | Hyperparameter optimization |
| `shap` | Model explainability |
| `ipykernel` | Notebook execution environment |

See `requirements.txt` for pinned dependency versions.

## Limitations

This project has the following limitations:

- Only CatBoost is evaluated as the main model family.
- The OOF diagnostic is post-selection because hyperparameters are selected before the OOF loop.
- Feature importance and SHAP values explain fitted model behavior, but do not prove causal effects.
- No controlled feature ablation study is included, so individual feature contributions are not isolated.
- High-cardinality variables such as `Ticket` and `Name` may capture dataset-specific structure.

## Next Steps

Potential extensions include:

- Run controlled feature ablation studies.
- Compare additional model families, such as logistic regression, random forest, XGBoost, LightGBM, and calibrated ensembles.
- Evaluate repeated cross-validation or nested model-selection protocols.
- Save experiment metadata alongside each generated submission.

## Author

Developed by **Giovanni Persio**.

This project is part of a data science portfolio focused on tabular machine learning, feature engineering, validation methodology, and model explainability.
