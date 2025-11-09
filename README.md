# Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI

This repository contains the reference implementation for the paper **"Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI"** published in the *International Journal of Approximate Reasoning*.

## Overview

This implementation demonstrates multiple aggregation strategies for SHAP (SHapley Additive exPlanations) values to create interpretable network representations of machine learning models. The approach aggregates individual-level explanations into population-level patterns while preserving uncertainty and heterogeneity in feature importance across different patient subgroups.

### Key Contributions

- **Robust Aggregation with Uncertainty Quantification**: Median-based aggregation with IQR-inspired percentile bounds for identifying consistent vs. uncertain feature effects
- **Phenotype Discovery**: Clustering-based aggregation to reveal distinct disease mechanisms across patient subgroups
- **Risk Stratification**: Comparative network analysis across risk levels to understand differential feature importance
- **Hypothesis-Driven Segmentation**: Manual segmentation for targeted subgroup analysis

## Repository Structure

```
IJAR-python/
├── run/                          # Jupyter notebooks implementing aggregation methods
│   ├── shap_network_median_iqr.ipynb           # Median + IQR aggregation
│   ├── shap_network_clustering.ipynb           # Clustering-based aggregation
│   ├── shap_network_risk_stratified.ipynb      # Risk-stratified aggregation
│   └── shap_network_manual_segmentation.ipynb  # Manual segmentation
├── data/                         # Pre-computed SHAP values and features
│   ├── shap_values.npy          # SHAP values for 500 patients
│   ├── shap_interaction_values.npy  # SHAP interaction values
│   └── x_values.pkl             # Feature matrix (NHANES dataset)
├── result/                       # Generated visualizations (HTML/PDF)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment support (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/seccijr/IJAR-python.git
   cd IJAR-python
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv

   # On Windows (Git Bash/MSYS):
   source .venv/Scripts/activate

   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Aggregation Methods

This implementation provides four distinct aggregation strategies for creating network representations from individual SHAP explanations:

### 1. Median + IQR Aggregation (`shap_network_median_iqr.ipynb`)

**Purpose**: Robust aggregation with uncertainty quantification using percentile-based bounds.

**Methodology**:
- Uses **median** instead of mean for robust central tendency (resistant to outliers)
- Implements a configurable tolerance parameter **γ ∈ [0, 50]** for percentile bounds:
  - Computes (50-γ)-th and (50+γ)-th percentiles for each feature
  - When γ = 25, recovers classical IQR (25th and 75th percentiles)

**Classification Scheme**:
- **Positive (risk) nodes**: Both percentiles positive → at least (50+γ)% of patients show positive effect
- **Negative (protective) nodes**: Both percentiles negative → at least (50+γ)% of patients show negative effect
- **Gray (mixed-sign) nodes**: Percentiles have different signs → high heterogeneity across patients

**Key Parameters**:
```python
gamma = 15  # Tolerance parameter (smaller = more permissive, larger = stricter)
```

**Equations**:

Node weights:
```
N*_i = median(Sh_i(Δ^t)) / Σ_u |median(Sh_u(Δ^t))|
```

Edge weights:
```
E*_ij = median(I_ij(Δ^t)) / Σ_v Σ_{u>v} |median(I_uv(Δ^t))|
```

### 2. Clustering-Based Aggregation (`shap_network_clustering.ipynb`)

**Purpose**: Discover patient phenotypes with distinct disease mechanisms through unsupervised learning.

**Methodology**:
- Performs K-means clustering on SHAP value patterns
- Evaluates optimal number of clusters using:
  - Elbow method (inertia)
  - Silhouette score (higher is better)
  - Davies-Bouldin score (lower is better)
- Builds separate networks for each discovered cluster

**Key Features**:
- Automatically identifies distinct patient subgroups
- Reveals cluster-specific feature importance patterns
- Provides comparative analysis of top features across clusters

**Equations**:

Node weights for cluster C_c:
```
N*_i = Σ_{t∈C_c} Sh_i(Δ^t) / Σ_u Σ_{t∈C_c} |Sh_u(Δ^t)|
```

Edge weights for cluster C_c:
```
E*_ij = Σ_{t∈C_c} I_ij(Δ^t) / Σ_v Σ_{u>v} Σ_{t∈C_c} |I_uv(Δ^t)|
```

where **u > v** denotes lower triangular matrix elements (excluding diagonal).

### 3. Risk-Stratified Aggregation (`shap_network_risk_stratified.ipynb`)

**Purpose**: Compare feature importance patterns across different risk levels.

**Methodology**:
- Stratifies patients by outcome variable (y) into risk groups:
  - **Low risk**: Bottom 66% of patients by outcome value
  - **High risk**: Top 34% of patients by outcome value
- Builds separate networks for each risk stratum
- Enables comparison of disease mechanisms across risk levels

**Use Cases**:
- Understanding differential effects in high-risk populations
- Identifying risk-specific biomarkers
- Personalizing interventions based on risk profile

**Equations**: Same sum-based aggregation as clustering, applied to each risk stratum R_k.

### 4. Manual Segmentation (`shap_network_manual_segmentation.ipynb`)

**Purpose**: Hypothesis-driven subgroup analysis based on specific predictor variables.

**Methodology**:
- Segments cohort using explicit clinical criteria
- Example: Males over 50 years old (age > 50 AND sex_isFemale == False)
- Applies same sum-based aggregation to each segment
- Enables targeted analysis of clinically relevant subgroups

**Use Cases**:
- Testing specific clinical hypotheses
- Regulatory subgroup analysis
- Population-specific model interpretation

**Equations**: Same sum-based aggregation as clustering and risk stratification.

## Usage

Each Jupyter notebook is self-contained and can be run independently:

```bash
# Start Jupyter
jupyter notebook

# Navigate to run/ directory and open desired notebook
```

### Example Workflow

1. **Open `run/shap_network_median_iqr.ipynb`**
2. **Configure parameters** (e.g., γ tolerance):
   ```python
   gamma = 15  # Adjust between 0-50
   ```
3. **Run all cells** to generate network visualization
4. **Experiment** with different γ values to see how uncertainty affects classification

### Visualization Parameters

All notebooks use customizable visualization parameters:

```python
positive_alpha = 0.01   # Threshold for positive edges
negative_alpha = 0.01   # Threshold for negative edges
positive_beta = 0       # Minimum threshold for positive nodes
negative_beta = 0       # Minimum threshold for negative nodes
```

## Data Description

The implementation uses pre-computed SHAP values from the **NHANES I** epidemiologic follow-up dataset:

- **Dataset**: National Health and Nutrition Examination Survey I
- **Samples**: 500 patients
- **Features**: 80 clinical and demographic variables
- **Target**: Disease outcome/survival
- **Model**: Pre-trained XGBoost survival model

### Data Files

| File | Description | Shape |
|------|-------------|-------|
| `x_values.pkl` | Feature matrix (clinical variables) | (500, 80) |
| `shap_values.npy` | Individual SHAP values | (500, 80) |
| `shap_interaction_values.npy` | SHAP interaction values | (500, 80, 80) |

### Key Features

The dataset includes clinical measurements such as:
- Demographics: age, sex
- Vital signs: systolic blood pressure, BMI, pulse pressure
- Laboratory values: serum albumin, cholesterol, hemoglobin, white blood cells, red blood cells
- Metabolic markers: uric acid, alkaline phosphatase, SGOT
- Other: physical activity, sedimentation rate

## Mathematical Framework

### Node Weights (N*_i)

Node weights represent the aggregated importance of each feature:

**Sum-based** (Clustering, Risk, Manual):
```
N*_i = Σ_t Sh_i(Δ^t) / Σ_u Σ_t |Sh_u(Δ^t)|
```

**Median-based** (IQR):
```
N*_i = median_t(Sh_i(Δ^t)) / Σ_u |median_t(Sh_u(Δ^t))|
```

### Edge Weights (E*_ij)

Edge weights represent feature interactions:

**Sum-based**:
```
E*_ij = Σ_t I_ij(Δ^t) / Σ_v Σ_{u>v} Σ_t |I_uv(Δ^t)|
```

**Median-based**:
```
E*_ij = median_t(I_ij(Δ^t)) / Σ_v Σ_{u>v} |median_t(I_uv(Δ^t))|
```

**Important**: The denominator sums only over the **lower triangular** matrix (u > v), excluding diagonal elements.

### Uncertainty Quantification (IQR Method)

For the median-based approach, uncertainty is quantified using percentile bounds:

```
Q_lower = percentile(Sh_i(Δ^t), 50 - γ)
Q_upper = percentile(Sh_i(Δ^t), 50 + γ)
IQR = Q_upper - Q_lower
```

Classification:
- **Positive**: Q_lower > 0 and Q_upper > 0
- **Negative**: Q_lower < 0 and Q_upper < 0
- **Uncertain**: sign(Q_lower) ≠ sign(Q_upper)

## Dependencies

Core dependencies (see `requirements.txt` for complete list):

- **shap**: SHAP value computation and visualization
- **xgboost**: Gradient boosting model (used for SHAP value generation)
- **scikit-learn**: Clustering, PCA, and preprocessing
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **networkx**: Network graph operations
- **cgt_perezsechi**: Custom cooperative game theory library for graph visualization and normalization

The `cgt_perezsechi` package provides specialized functionality for:
- Graph visualization (`cgt_perezsechi.visualization.graph.draw`)
- Matrix normalization (`cgt_perezsechi.manipulation.norm.normalize_psi`, `normalize_r`)

## Output Files

Generated visualizations are saved in the `result/` directory:

- **HTML files**: Interactive visualizations viewable in web browser
- **PDF files**: Publication-ready network diagrams

Example outputs:
- `shap_network_median_iqr.pdf`
- `shap_network_clustering.pdf`
- `shap_network_risk_stratified.pdf`

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{fuzzy_xai_ijar,
  title={Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI},
  author={[Authors]},
  journal={International Journal of Approximate Reasoning},
  year={2025},
  note={Reference implementation: https://github.com/seccijr/IJAR-python}
}
```

## Related Resources

- **Paper**: "Fuzzy Modeling to Explanation: Aggregating Multi-Measures Fuzzy Systems for XAI" (International Journal of Approximate Reasoning)
- **CGT Library**: [https://github.com/perez-sechi/cgt](https://github.com/perez-sechi/cgt) - Cooperative Game Theory library for visualization
- **SHAP**: [https://github.com/slundberg/shap](https://github.com/slundberg/shap) - SHapley Additive exPlanations
- **NHANES**: [https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/) - National Health and Nutrition Examination Survey

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contact

For questions or issues regarding this implementation, please open an issue on the [GitHub repository](https://github.com/seccijr/IJAR-python/issues).

## Acknowledgments

This work uses the NHANES I epidemiologic follow-up dataset and builds upon the SHAP framework for model interpretability. The network visualization capabilities are provided by the cgt_perezsechi library.
