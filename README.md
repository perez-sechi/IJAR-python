# IJAR-python

Computational experiments and network visualizations supporting the paper:

> **"From Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI"**
> Carlos I. Pérez-Sechi, Inmaculada Gutiérrez, Javier Castro, Daniel Gómez, Daniel Martín, Rosa Espínola
> *International Journal of Approximate Reasoning*, 2026

This repository contains the complete Python implementation of the **Multi-Measure Fuzzy System (MMFS)** aggregation methodology proposed in the paper. It reproduces every figure in the manuscript and extends the analysis to additional datasets and model families beyond those discussed in the paper.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Mapping to the Manuscript](#mapping-to-the-manuscript)
  - [Theoretical Example (Example 2 / Section 4)](#theoretical-example-example-2--section-4)
  - [Computation Notebooks (Section 7)](#computation-notebooks-section-7)
  - [Visualization Notebooks (Section 7)](#visualization-notebooks-section-7)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Dependencies](#dependencies)

---

## Overview

The paper introduces a framework that interprets machine learning predictions as a **Multi-Measure Fuzzy System** (Definition 4 in the paper), where each instance $k$ of a dataset $\mathcal{D}$ defines a generalized fuzzy measure via the variation of prediction $\Delta^k$ (Definition 5, Eq. 1). The codebase operationalizes the full pipeline from Algorithm 1 (MMFS-to-Graph Pipeline):

1. **Representation** — Each fuzzy measure $\mu^k = \Delta^k$ is reduced from its exponential domain $\mathcal{P}(S)$ to a compact tensor space via a representation function $\mathcal{R}_p$ (Definition 3):
   - $p=1$: MMFS relevance representation vectors $v^k = \mathcal{R}_1^k(\Delta^k) = Sh(\Delta^k)$ — the SHAP values (Definition 6, Eq. 2).
   - $p=2$: MMFS interactions representation matrices $M^k = \mathcal{R}_2^k(\Delta^k) = I(\Delta^k)$ — the SHAP interaction values (Definition 8, Eq. 3).

2. **Aggregation** — Node weighting vectors $\mathcal{N}^*_i$ (Definition 7) and edge weighting matrices $\mathcal{E}^*_{ij}$ (Definition 9) aggregate these representations across instances using five distinct strategies (Sections 7.1–7.5), each resolving the $\mathcal{N}^*_i$ / $\mathcal{E}^*_{ij}$ functions differently.

3. **Visualization** — The aggregated weights define the nodes and edges of an interpretable network graph $G=(V,E)$, where node size encodes feature importance and edge width encodes feature-pair interaction strength.

---

## Project Structure

```
IJAR-python/
├── requirements.txt                         # Python dependencies
├── data/                                    # Pre-computed SHAP values and datasets
│   ├── credit/                              # German Credit dataset
│   │   ├── x_values.pkl                     # Feature matrix (500 samples, 48 one-hot encoded features)
│   │   ├── y_values.pkl                     # Target variable (good/bad credit)
│   │   ├── rf/                              # Random Forest SHAP outputs
│   │   │   ├── shap_values.npy              # Shape: (500, 48) — Sh_i(Δ^k) for the positive class
│   │   │   └── shap_interaction_values.npy  # Shape: (500, 48, 48) — I_ij(Δ^k) for the positive class
│   │   └── xgboost/                         # XGBoost SHAP outputs
│   │       ├── shap_values.npy
│   │       └── shap_interaction_values.npy
│   └── nhanesi/                             # NHANES I dataset
│       ├── x_values.pkl                     # Feature matrix (500 samples, 79 health variables)
│       ├── rf/
│       │   ├── shap_values.npy              # Shape: (500, 79)
│       │   └── shap_interaction_values.npy  # Shape: (500, 79, 79)
│       └── xgboost/
│           ├── shap_values.npy
│           └── shap_interaction_values.npy
├── result/                                  # Generated network visualizations (.jpg)
└── run/
    ├── computation/                         # Model training & SHAP computation
    │   ├── example_2_shapley_grabisch.ipynb
    │   ├── nhanesi_xgboost_shap.ipynb
    │   ├── nhanesi_rf_shap.ipynb
    │   ├── credit_xgboost_shap.ipynb
    │   └── credit_rf_shap.ipynb
    └── visualization/                       # 5 aggregation strategies × 4 dataset/model combinations
        ├── credit_rf_global_mean_network.ipynb
        ├── credit_rf_risk_stratified_network.ipynb
        ├── credit_rf_clustering_network.ipynb
        ├── credit_rf_manual_segmentation_network.ipynb
        ├── credit_rf_median_iqr_network.ipynb
        ├── credit_xgboost_global_mean_network.ipynb
        ├── credit_xgboost_risk_stratified_network.ipynb
        ├── credit_xgboost_clustering_network.ipynb
        ├── credit_xgboost_manual_segmentation_network.ipynb
        ├── credit_xgboost_median_iqr_network.ipynb
        ├── nhanesi_rf_global_mean_network.ipynb
        ├── nhanesi_rf_risk_stratified_network.ipynb
        ├── nhanesi_rf_clustering_network.ipynb
        ├── nhanesi_rf_manual_segmentation_network.ipynb
        ├── nhanesi_rf_median_iqr_network.ipynb
        ├── nhanesi_xgboost_global_mean_network.ipynb
        ├── nhanesi_xgboost_risk_stratified_network.ipynb
        ├── nhanesi_xgboost_clustering_network.ipynb
        ├── nhanesi_xgboost_manual_segmentation_network.ipynb
        └── nhanesi_xgboost_median_iqr_network.ipynb
```

---

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

The key dependency [`cgt_perezsechi`](https://github.com/perez-sechi/cgt) is a cooperative game theory library (installed directly from GitHub) that provides:

- Exact Shapley value computation (`cgt_perezsechi.compute.shapley.exact`) — implements Definition 2 (Eq. 1) of the paper.
- Grabisch interaction index computation (`cgt_perezsechi.compute.grabisch`) — implements Definition 3 (Eq. 2) of the paper.
- Network graph drawing and normalization utilities — used by all visualization notebooks.

---

## Datasets

### NHANES I (primary example in the paper)

The National Health and Nutrition Examination Survey I dataset, loaded via `shap.datasets.nhanesi()` without modification (no feature selection, normalization, or imputation, as stated in Section 7 of the paper). It contains health and survival data for predicting the long-term probability of death, with **79 predictor variables** — referred to as *agents* $S = \{X_1, \dots, X_{79}\}$ in the MMFS framework. SHAP values and interaction values are computed on the first **500 instances**.

This dataset is the primary subject of all figures (Figures 1–7) in the manuscript.

### German Credit

The German Credit dataset (`credit-g` from OpenML) for predicting credit risk (good/bad). Categorical features are one-hot encoded, yielding **48 features**. SHAP values and interaction values are computed on the first **500 instances** (positive class only). This dataset extends the analysis beyond the paper's main examples, demonstrating the generality of the MMFS framework.

---

## Mapping to the Manuscript

### Theoretical Example (Example 2 / Section 4)

**Notebook:** `run/computation/example_2_shapley_grabisch.ipynb`

This notebook implements **Example 2** (Section 4) of the paper, which illustrates the core mathematical machinery without any machine learning model. It is the computational counterpart to the hand-computed results in Examples 1–3 and Table 1.

The notebook defines the MMFS $\mathcal{MF} = (S, \mathcal{F})$ with $S = \{1, 2, 3\}$ and two expert fuzzy measures $\mu_1, \mu_2$ from Table 1:

| $A \subseteq S$ | $\emptyset$ | $\{1\}$ | $\{2\}$ | $\{3\}$ | $\{1,2\}$ | $\{1,3\}$ | $\{2,3\}$ | $\{1,2,3\}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\mu_1(A)$ | 0 | 0.50 | 0.30 | 0.40 | 0.60 | 0.80 | 0.70 | 1.00 |
| $\mu_2(A)$ | 0 | 0.20 | 0.10 | 0.20 | 0.30 | 0.40 | 0.35 | 1.00 |

It then:

- Computes the **MMFS relevance representation vectors** $v^k = \mathcal{R}_1^k(\mu^k) = Sh(\mu^k)$ (Definition 6) using `cgt_perezsechi.compute.shapley.exact`, yielding $v^1 = (0.3833, 0.2333, 0.3833)$ and $v^2 = (0.35, 0.275, 0.375)$.
- Computes the **node weighting vector** $v_i^* = \mathcal{N}^*_i(v^1, v^2) = \text{mean}(Sh_i(\mu^1), Sh_i(\mu^2))$ (Definition 7), yielding $v^* = (0.3667, 0.2542, 0.3792)$.
- Computes the **MMFS interactions representation matrices** $M^k = \mathcal{R}_2^k(\mu^k) = I(\mu^k)$ (Definition 8) using `cgt_perezsechi.compute.grabisch`.
- Computes the **edge weighting matrix** $e_{ij}^* = \mathcal{E}^*_{ij}(M^1, M^2) = \text{mean}(I_{ij}(\mu^1), I_{ij}(\mu^2))$ (Definition 9), yielding the aggregated $M^*$ shown in Example 3.

This is the only notebook that uses exact symbolic fuzzy measures rather than SHAP-derived ones.

---

### Computation Notebooks (Section 7)

Located in `run/computation/`, these notebooks train machine learning models and compute the SHAP values that serve as input to the visualization pipeline. They materialize the MMFS $\mathcal{MF} = (S, \mathcal{F})$ with $\mathcal{F} = \{\Delta^1, \dots, \Delta^m\}$, where each $\Delta^k$ is the variation of prediction (Definition 5) for instance $k$, and the SHAP values $Sh_i(\Delta^k)$ and interaction indices $I_{ij}(\Delta^k)$ correspond to the MMFS representation vectors and matrices (Eqs. 2–3).

#### `nhanesi_xgboost_shap.ipynb` — Primary model in the paper

Trains the XGBoost survival model (`survival:cox` objective, learning rate 0.002, max depth 3, subsampling 0.5, 5000 boosting rounds) analyzed throughout Section 7. Computes SHAP values and SHAP interaction values for 500 instances via `shap.TreeExplainer`. Any `NaN` entries in the interaction matrices are set to zero prior to aggregation, as noted in Section 7 of the paper. Outputs saved to `data/nhanesi/xgboost/`.

#### `nhanesi_rf_shap.ipynb`

Trains a Random Forest regressor (500 trees, max depth 6, min samples per leaf 5) on NHANES I. Computes SHAP values and interaction values for 500 instances. Outputs saved to `data/nhanesi/rf/`. Provides an alternative model perspective for the same health dataset.

#### `credit_rf_shap.ipynb`

Trains a Random Forest classifier (500 trees, max depth 6, min samples per leaf 5) on the German Credit dataset. Computes SHAP values and interaction values for 500 instances (positive class). Outputs saved to `data/credit/rf/`.

#### `credit_xgboost_shap.ipynb`

Trains an XGBoost classifier on the German Credit dataset. Computes SHAP values and interaction values for 500 instances (positive class). Outputs saved to `data/credit/xgboost/`.

---

### Visualization Notebooks (Section 7)

All visualization notebooks implement **Algorithm 1** (MMFS-to-Graph Pipeline) from the paper. They load pre-computed SHAP values and interaction values from `data/`, define specific instances of $\mathcal{N}^*_i$ and $\mathcal{E}^*_{ij}$, construct weighted graphs via `cgt_perezsechi`, and save network visualizations to `result/`.

All networks use a consistent visual encoding (introduced in Section 7): **blue** nodes/edges indicate risk factors (positive contributions), **red** nodes/edges indicate protective factors (negative contributions), and **gray** encodes uncertainty. Node size and edge width are proportional to the normalized absolute value of the respective representation function.

The visualization notebooks are provided for all four dataset/model combinations (NHANES I × {XGBoost, RF} and German Credit × {XGBoost, RF}). The paper's figures are generated from the **NHANES I / XGBoost** combination.

---

#### Global Mean Summarization — `*_global_mean_network.ipynb`

**Paper:** Section 7.1 — **Figure 1**

Implements the baseline aggregation strategy by computing a single global network over all $m = 500$ instances. The node weighting vector $\mathcal{N}^*_i$ (Eq. 7) and edge weighting matrix $\mathcal{E}^*_{ij}$ (Eq. 8) are defined as normalized absolute sums:

$$
\mathcal{N}^*_i = \frac{\sum_{k=1}^{m} |Sh_i(\Delta^k)|}{\sum_{u=1}^{n} \sum_{k=1}^{m} |Sh_u(\Delta^k)|}
\qquad
\mathcal{E}^*_{ij} = \frac{\sum_{k=1}^{m} |I_{ij}(\Delta^k)|}{\sum_{v=1}^{n} \sum_{u > v}^{n} \sum_{k=1}^{m} |I_{uv}(\Delta^k)|}
$$

The denominator in $\mathcal{E}^*_{ij}$ sums only the lower-triangular portion of the interaction matrix (excluding the diagonal), consistent with the paper's specification. This corresponds to the standard global SHAP summary used by XAI libraries. The resulting network (Figure 1) shows `age` as the most central node, with strong interactions with `sex_isFemale` and `systolic_blood_pressure`.

---

#### Risk-Stratified Summarization — `*_risk_stratified_network.ipynb`

**Paper:** Section 7.2 — **Figure 2**

Stratifies the 500 instances by predicted mortality risk into a Low Risk stratum $P^1$ (bottom 67%, 330 instances) and a High Risk stratum $P^2$ (top 33%, 170 instances). A separate network is computed for each stratum by restricting the summation in $\mathcal{N}^*_i$ and $\mathcal{E}^*_{ij}$ (Eqs. 9–10) to instances $k \in P^\ell$.

The resulting pair of networks (Figure 2) reveals group-specific drivers: in the Low Risk stratum, `sex_isFemale` and `systolic_blood_pressure` dominate; in the High Risk stratum, inflammatory markers (`sedimentation_rate`), `bmi`, and `serum_albumin` emerge as key factors.

---

#### Clustering-Based Summarization — `*_clustering_network.ipynb`

**Paper:** Section 7.3 — **Figure 3**

Partitions instances into latent phenotypes by applying K-Means clustering directly on the standardized MMFS relevance representation vectors $v^k = Sh(\Delta^k)$. The optimal number of clusters is selected by evaluating inertia, silhouette scores, and Davies-Bouldin indices (the paper identifies $k=4$ clusters for NHANES I / XGBoost: $|C^0|=10$, $|C^1|=250$, $|C^2|=24$, $|C^3|=216$). A separate network is then computed for each cluster $C^\ell$ using the same formulas (Eqs. 11–12) restricted to cluster members.

The four-panel Figure 3 in the paper shows that features such as `systolic_blood_pressure`, `serum_albumin`, and `sedimentation_rate` have high variance across phenotypes, confirming their cluster-specific importance.

---

#### Manual Segmentation — `*_manual_segmentation_network.ipynb`

**Paper:** Section 7.4 — **Figures 4 and 5**

Applies hypothesis-driven partitioning of the cohort using domain-specific predictor rules:

- **NHANES I**: Males over 50 ($\texttt{age} > 50$ and $\texttt{sex\_isFemale} = \text{false}$), giving $|T^1| = 111$ (22.2%) vs. $|T^2| = 389$ (77.8%). Formally: $T^1 = \{k \mid \text{age}^{(k)} > 50 \wedge \text{sex\_isFemale}^{(k)} = \text{false}\}$.
- **German Credit**: Long-duration debtors with no checking account ($\texttt{duration} > 30$ and $\texttt{checking\_status\_<0} = \text{true}$) vs. the rest.

The same aggregation functions (Eqs. 13–14) are applied to each manually defined segment $T^\ell$. Figure 4 shows `age` as a risk factor (blue) for males over 50, while Figure 5 shows the same feature as protective (red) for the rest — demonstrating opposite directionality depending on population composition, as discussed in Section 7.4.

---

#### Robust Summarization (Median + IQR) — `*_median_iqr_network.ipynb`

**Paper:** Section 7.5 — **Figures 6 and 7**

Replaces the mean with the **median** ($P_{50}$) in both the node and edge representation functions (Eqs. 15–16) for robustness to outlier instances:

$$
\mathcal{N}^*_i = \frac{P_{50}(|Sh_i(\Delta^1)|, \dots, |Sh_i(\Delta^m)|)}{\sum_{u=1}^{n} P_{50}(|Sh_u(\Delta^1)|, \dots, |Sh_u(\Delta^m)|)}
$$

Additionally introduces a **tolerance parameter** $\gamma \in [0, 50]$ to quantify sign consistency via percentile bounds (Eqs. 17–18):

$$
Q^{(i)}_\text{lower} = P_{50-\gamma}(Sh_i(\Delta^1), \dots, Sh_i(\Delta^m)), \quad Q^{(i)}_\text{upper} = P_{50+\gamma}(Sh_i(\Delta^1), \dots, Sh_i(\Delta^m))
$$

Nodes and edges are classified by sign consistency:
- **Positive (risk):** both bounds $> 0$ — at least $(50+\gamma)\%$ of instances agree on positive direction.
- **Negative (protective):** both bounds $< 0$.
- **Gray (uncertain):** bounds cross zero — mixed directional effects.

The notebook uses $\gamma = 15$ (Figure 6, 35th–65th percentile band) as the primary setting, and produces additional sensitivity figures for $\gamma = 5$ and $\gamma = 25$ (Figure 7). It also sweeps $\gamma$ from 1 to 49 to compute the critical $\gamma^*$ at which each variable first transitions to gray, as listed in Section 7.5.

---

## Reproducing the Paper Results

Run the experiments in two phases:

### Phase 1: Compute SHAP values (if not using pre-computed data)

Pre-computed `.npy` files are already included in `data/`. Run the computation notebooks only if you want to retrain the models from scratch.

```
run/computation/nhanesi_xgboost_shap.ipynb   # Primary model in the paper (computationally intensive)
run/computation/nhanesi_rf_shap.ipynb
run/computation/credit_rf_shap.ipynb
run/computation/credit_xgboost_shap.ipynb
run/computation/example_2_shapley_grabisch.ipynb  # Theoretical example (fast)
```

### Phase 2: Generate network visualizations

The paper's figures are generated from the **NHANES I / XGBoost** combination:

| Paper Reference | Notebook |
|---|---|
| Figure 1 — Section 7.1 (Eq. 7–8, global mean) | `nhanesi_xgboost_global_mean_network.ipynb` |
| Figure 2 — Section 7.2 (Eq. 9–10, risk-stratified) | `nhanesi_xgboost_risk_stratified_network.ipynb` |
| Figure 3 — Section 7.3 (Eq. 11–12, clustering, $k=4$) | `nhanesi_xgboost_clustering_network.ipynb` |
| Figures 4–5 — Section 7.4 (Eq. 13–14, manual segmentation) | `nhanesi_xgboost_manual_segmentation_network.ipynb` |
| Figures 6–7 — Section 7.5 (Eq. 15–18, median + IQR, $\gamma \in \{5,15,25\}$) | `nhanesi_xgboost_median_iqr_network.ipynb` |
| Example 2 — Section 4 (Table 1, Definitions 6–9) | `computation/example_2_shapley_grabisch.ipynb` |

The same five visualization notebooks are also provided for NHANES I / Random Forest, German Credit / Random Forest, and German Credit / XGBoost, extending the analysis beyond the paper's primary examples.

---

## Dependencies

| Package | Purpose |
|---|---|
| [`cgt_perezsechi`](https://github.com/perez-sechi/cgt) | Cooperative Game Theory: exact Shapley values (Def. 2), Grabisch interaction indices (Def. 3), graph drawing and normalization |
| `shap` | SHAP values $Sh_i(\Delta^k)$ and interaction values $I_{ij}(\Delta^k)$ via `TreeExplainer` |
| `xgboost` | XGBoost survival and classification models |
| `scikit-learn` | Random Forest models, K-Means clustering (Section 7.3), `StandardScaler` |
| `numpy` | Numerical computation and `.npy` file I/O |
| `matplotlib` | Network graph rendering |
| `networkx` | Graph data structures |
| `seaborn` | Color maps for positive/negative/gray encoding |
| `statsmodels`, `pingouin` | Statistical utilities |

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for full details.

---

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
