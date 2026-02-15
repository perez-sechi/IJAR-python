# IJAR-python

Experimental codebase supporting the paper **"From Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI"**, published in the _International Journal of Approximate Reasoning_.

This repository contains all the computational experiments and network visualizations presented in the paper. It implements the proposed Multi-Measure Fuzzy System (MMFS) aggregation methodology, which summarizes SHAP-based fuzzy measures into interpretable graph representations for Explainable Artificial Intelligence (XAI).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [Computation Notebooks](#computation-notebooks)
  - [Visualization Notebooks](#visualization-notebooks)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Dependencies](#dependencies)

## Overview

The paper proposes a framework that:

1. Interprets machine learning models as Multi-Measure Fuzzy Systems (MMFS), where each instance defines a fuzzy measure via the SHAP methodology.
2. Introduces **representation functions** that reduce fuzzy measures from $\mathcal{P}(S)$ (dimension $2^S$) to tensor spaces — specifically, the Shapley value ($p=1$, vectors) and the interaction index ($p=2$, matrices).
3. Aggregates these representations across instances using node weighting vectors ($\mathcal{N}^*_i$) and edge weighting matrices ($\mathcal{E}^*_{ij}$) to produce interpretable network graphs.

This codebase provides the complete pipeline: from training models and computing SHAP values, through applying the different aggregation strategies presented in the paper, to generating the network visualizations that appear in the paper's figures.

## Project Structure

```
IJAR-python/
├── requirements.txt                    # Python dependencies
├── data/                               # Pre-computed data and SHAP values
│   ├── credit/                         # German Credit dataset
│   │   ├── x_values.pkl                # Feature matrix
│   │   ├── y_values.pkl                # Target variable
│   │   ├── rf/                         # Random Forest SHAP outputs
│   │   │   ├── shap_values.npy
│   │   │   └── shap_interaction_values.npy
│   │   └── xgboost/                    # XGBoost SHAP outputs
│   │       ├── shap_values.npy
│   │       └── shap_interaction_values.npy
│   └── nhanesi/                        # NHANES I dataset
│       ├── x_values.pkl
│       ├── rf/
│       │   ├── shap_values.npy
│       │   └── shap_interaction_values.npy
│       └── xgboost/
│           ├── shap_values.npy
│           └── shap_interaction_values.npy
├── result/                             # Generated network visualizations (.jpg)
└── run/
    ├── computation/                    # Model training & SHAP computation
    │   ├── example_2_shapley_grabisch.ipynb
    │   ├── nhanesi_xgboost_shap.ipynb
    │   ├── nhanesi_rf_shap.ipynb
    │   ├── credit_xgboost_shap.ipynb
    │   └── credit_rf_shap.ipynb
    └── visualization/                  # Network graph generation
        ├── credit/
        │   ├── rf/
        │   │   ├── global_mean_network.ipynb
        │   │   ├── risk_stratified_network.ipynb
        │   │   ├── clustering_network.ipynb
        │   │   ├── manual_segmentation_network.ipynb
        │   │   └── median_iqr_network.ipynb
        │   └── xgboost/
        │       └── (same notebooks as rf/)
        └── nhanesi/
            ├── rf/
            │   └── (same notebooks as above)
            └── xgboost/
                └── (same notebooks as above)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/perez-sechi/IJAR-python.git
cd IJAR-python

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

The key dependency [`cgt_perezsechi`](https://github.com/perez-sechi/cgt) is a cooperative game theory library that provides:

- Exact Shapley value computation
- Grabisch interaction index computation
- Network graph drawing and normalization utilities

## Datasets

### NHANES I

The National Health and Nutrition Examination Survey I (NHANES I) dataset, loaded via `shap.datasets.nhanesi()`, is used as the primary example in the paper (Section 7). It contains health and survival data for predicting long-term probability of death, with 79 predictor variables including demographics, lifestyle habits, and serum biomarkers.

### German Credit

The German Credit dataset (`credit-g` from OpenML) is a classification task for predicting credit risk (`good`/`bad`). Categorical features are one-hot encoded, yielding a high-dimensional feature space. This dataset provides a complementary use case beyond the NHANES I example discussed in the paper, demonstrating the generality of the MMFS framework.

## Experiments

### Computation Notebooks

Located in `run/computation/`, these notebooks train machine learning models on each dataset and compute the SHAP values and interaction values that serve as input to the visualization pipeline. The SHAP values correspond to the Shapley indices $Sh_i(\Delta^k)$ of the fuzzy measures, and the SHAP interaction values correspond to the interaction indices $I_{ij}(\Delta^k)$, as described in the paper's Section 6.

#### Theoretical Example — `example_2_shapley_grabisch.ipynb`

Implements **Example 2** from the paper (Section 4), where fuzzy measures $\mu_1$ and $\mu_2$ on a set of 3 system components $S = \{1, 2, 3\}$ are analyzed using two experts' perspectives. This notebook:

- Defines the fuzzy measures from the risk analysis example (Table 1 in the paper)
- Computes exact Shapley values using `cgt_perezsechi.compute.shapley.exact`, yielding the MMFS relevance representation vectors $\mathcal{R}_1^k(\mu^k) = Sh(\mu^k)$
- Computes Grabisch interaction indices for all pairs $(i, j)$ using `cgt_perezsechi.compute.grabisch`, yielding the MMFS interactions representation matrices $\mathcal{R}_2^k(\mu^k) = I(\mu^k)$
- Aggregates these across the two experts by averaging, demonstrating the node weighting vector $\mathcal{N}^*_i$ and edge weighting matrix $\mathcal{E}^*_{ij}$ definitions

This notebook illustrates the core mathematical concepts without any machine learning model.

#### NHANES I — XGBoost — `nhanesi_xgboost_shap.ipynb`

Trains an XGBoost survival model (`survival:cox` objective) on the NHANES I dataset, which is the primary model analyzed in Section 7 of the paper. Configuration: learning rate 0.002, max depth 3, subsampling 0.5, 5000 boosting rounds. Computes SHAP values and SHAP interaction values for 500 instances using `shap.TreeExplainer`, saving them as `.npy` files for downstream visualization.

#### NHANES I — Random Forest — `nhanesi_rf_shap.ipynb`

Trains a Random Forest regressor (500 trees, max depth 6) on the NHANES I dataset and computes SHAP values and interaction values for 500 instances. Provides an alternative model perspective on the same health dataset.

#### German Credit — Random Forest — `credit_rf_shap.ipynb`

Trains a Random Forest classifier (500 trees, max depth 6) on the German Credit dataset. Computes SHAP values and interaction values for 500 instances (positive class).

#### German Credit — XGBoost — `credit_xgboost_shap.ipynb`

Trains an XGBoost model (`survival:cox` objective) on the German Credit dataset. Computes SHAP values and interaction values for 500 instances.

### Visualization Notebooks

Located in `run/visualization/{dataset}/{model}/`, these notebooks implement the five aggregation strategies from Section 7 of the paper. Each strategy defines specific node weighting vectors $\mathcal{N}^*_i$ and edge weighting matrices $\mathcal{E}^*_{ij}$ to aggregate the MMFS relevance representation vectors and interaction matrices into interpretable network graphs.

All notebooks load pre-computed SHAP values and interaction values from `data/`, construct weighted graphs using the `cgt_perezsechi` library, and save the resulting network visualizations to `result/`.

#### Global Mean Summarization — `global_mean_network.ipynb`

Implements **Section 7.1** of the paper. Computes a single global network by aggregating Shapley values and interaction indices across all 500 instances using the normalized mean. The node weighting vector (Eq. 7 in the paper) and edge weighting matrix (Eq. 8) are:

$$
\mathcal{N}^*_i = \frac{\sum_{k=1}^{m} |Sh_i(\Delta^k)|}{\sum_{u=1}^{n} \sum_{k=1}^{m} |Sh_u(\Delta^k)|}
\qquad
\mathcal{E}^*_{ij} = \frac{\sum_{k=1}^{m} |I_{ij}(\Delta^k)|}{\sum_{v=1}^{n} \sum_{u>v}^{n} \sum_{k=1}^{m} |I_{uv}(\Delta^k)|}
$$

This provides a baseline view of the model's overall behavior and corresponds to **Figure 1** in the paper (for NHANES I / XGBoost).

#### Risk-Stratified Summarization — `risk_stratified_network.ipynb`

Implements **Section 7.2** of the paper. Stratifies instances into low-risk (bottom 66%) and high-risk (top 34%) groups based on the model's output variable, then computes separate networks for each stratum using the same normalized sum aggregation restricted to each group. The formulas (Eq. 9–10) are identical to the global mean but sum only over instances in each stratum $P^\ell$.

This reveals group-specific mechanisms — for example, in NHANES I, the high-risk stratum highlights inflammatory markers and BMI as key drivers, while the low-risk stratum emphasizes sex and blood pressure. Corresponds to **Figure 2** in the paper (for NHANES I / XGBoost).

#### Clustering-Based Summarization — `clustering_network.ipynb`

Implements **Section 7.3** of the paper. Discovers latent subgroups by performing K-Means clustering on the standardized SHAP value vectors. The optimal number of clusters is determined by evaluating inertia, silhouette scores, and Davies-Bouldin indices. A separate network is then constructed for each cluster, using the same aggregation formulas (Eq. 11–12) restricted to cluster members.

This partitions instances into phenotypes with distinct explanation profiles, revealing heterogeneous mechanisms hidden by a global average. For NHANES I / XGBoost, the analysis identifies 4 clusters. Corresponds to **Figure 3** in the paper.

#### Manual Segmentation — `manual_segmentation_network.ipynb`

Implements **Section 7.4** of the paper. Allows hypothesis-driven partitioning of the cohort using domain-specific rules applied to predictor variables:

- **NHANES I**: Males over 50 (`age > 50` and `sex_isFemale = False`) vs. the rest of the cohort
- **German Credit**: Long-duration debtors with no checking account (`duration > 30` and `checking_status_<0 = True`) vs. the rest

The same aggregation formulas (Eq. 13–14) are applied to each manually defined segment, enabling comparative analysis. This demonstrates how the same features can have opposite directional effects depending on the population composition. Corresponds to **Figures 4 and 5** in the paper (for NHANES I / XGBoost).

#### Robust Summarization (Median + IQR) — `median_iqr_network.ipynb`

Implements **Section 7.5** of the paper. Replaces the mean with the **median** for both node and edge weights (Eq. 15–16), providing robustness to outlier instances. Additionally introduces a percentile-based uncertainty quantification mechanism through a tolerance parameter $\gamma = 15$:

- Computes the $(50-\gamma)$-th and $(50+\gamma)$-th percentile bounds for each feature and feature pair
- Classifies nodes and edges by sign consistency:
  - **Positive** (risk): both bounds > 0
  - **Negative** (protective): both bounds < 0
  - **Gray** (uncertain): bounds cross zero, indicating mixed effects across instances

This three-color encoding reveals which features have a consistent directional effect across the population. Corresponds to **Figure 6** in the paper (for NHANES I / XGBoost).

## Reproducing the Paper Results

The experiments should be run in two phases:

### Phase 1: Compute SHAP values (if not using pre-computed data)

Run the computation notebooks in `run/computation/`. These are computationally intensive (especially the interaction values) and their outputs are already included in `data/`.

```
run/computation/nhanesi_xgboost_shap.ipynb   # Primary model in the paper
run/computation/nhanesi_rf_shap.ipynb
run/computation/credit_rf_shap.ipynb
run/computation/credit_xgboost_shap.ipynb
run/computation/example_2_shapley_grabisch.ipynb  # Theoretical example (fast)
```

### Phase 2: Generate network visualizations

Run the visualization notebooks in `run/visualization/`. The paper's figures are generated primarily from the **NHANES I / XGBoost** combination:

| Paper Section                     | Paper Figure | Notebook                                            |
| --------------------------------- | ------------ | --------------------------------------------------- |
| Section 7.1 — Global Mean         | Figure 1     | `nhanesi/xgboost/global_mean_network.ipynb`         |
| Section 7.2 — Risk-Stratified     | Figure 2     | `nhanesi/xgboost/risk_stratified_network.ipynb`     |
| Section 7.3 — Clustering-Based    | Figure 3     | `nhanesi/xgboost/clustering_network.ipynb`          |
| Section 7.4 — Manual Segmentation | Figures 4, 5 | `nhanesi/xgboost/manual_segmentation_network.ipynb` |
| Section 7.5 — Median + IQR        | Figure 6     | `nhanesi/xgboost/median_iqr_network.ipynb`          |
| Section 4 — Theoretical Example   | Example 2    | `computation/example_2_shapley_grabisch.ipynb`      |

The same visualization notebooks are also provided for the NHANES I / Random Forest, German Credit / Random Forest, and German Credit / XGBoost combinations, extending the analysis beyond the examples presented in the paper.

## Dependencies

| Package                                                | Purpose                                                                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `shap`                                                 | SHAP values and interaction values via `TreeExplainer`                                                       |
| `xgboost`                                              | XGBoost survival models                                                                                      |
| `scikit-learn`                                         | Random Forest models, K-Means clustering, StandardScaler                                                     |
| `numpy`                                                | Numerical computation and `.npy` file I/O                                                                    |
| `matplotlib`                                           | Network graph rendering                                                                                      |
| `networkx`                                             | Graph data structures                                                                                        |
| `seaborn`                                              | Color maps for positive/negative encoding                                                                    |
| [`cgt_perezsechi`](https://github.com/perez-sechi/cgt) | Cooperative Game Theory: exact Shapley values, Grabisch interaction indices, graph drawing and normalization |

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for full details.

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@article{perezsechi2025fuzzy,
  title={From Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI},
  author={P{\'e}rez-Sechi, Carlos I. and Guti{\'e}rrez, Inmaculada and Castro, Javier and G{\'o}mez, Daniel and Mart{\'i}n, Daniel and Esp{\'i}nola, Rosa},
  journal={International Journal of Approximate Reasoning},
  year={2026}
}
```
