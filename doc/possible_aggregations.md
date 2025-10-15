## Stratified Aggregations (Subgroup Analysis)

By clinical/demographic subgroups:
- Age stratification (young/middle-aged/elderly)
- Risk level strata (based on predicted probability quartiles)
- Clinical phenotypes (hypertensive/normotensive, diabetic/non-diabetic)
- Gender, BMI categories, etc.

Implementation: Create network for each subgroup, compare architectures

## Robust Statistical Measures

Instead of mean/sum:
- Median aggregation - Robust to outliers, captures typical patient
- Trimmed mean - Remove top/bottom 10% before averaging
- Weighted mean - Weight by prediction confidence or disease probability
- Percentile-based (75th, 90th) - Focus on strong, consistent effects

## Direction-Aware Aggregations

Current approach mixes positive/negative effects:
- Signed magnitude - Preserve sign, sum only same-direction values
- Consensus aggregation - Only include if ≥70% of patients agree on direction
- Dominant direction mean - For each feature, take mean of dominant sign only
- Separate positive/negative networks - Two networks showing protective vs. risk factors

## Frequency-Based Aggregations

Magnitude × Frequency:
- Count how often feature is "important" (|SHAP| > threshold)
- Multiply average SHAP by frequency of importance
- Shows consistently relevant vs. occasionally extreme features

## Variance/Stability Measures

Capture heterogeneity:
- Coefficient of variation - Mean/StdDev ratio (relative consistency)
- Entropy-weighted - Higher weight to features with consistent effects
- Min-max range - Show range of effects across population

## Clustering-Based Aggregations

Patient phenotype discovery:
- Cluster patients by SHAP value patterns
- Build separate network for each cluster
- Reveals distinct disease mechanisms in subpopulations

## Outcome-Stratified Aggregations

Separate by prediction characteristics:
- High-risk patients (predicted probability > 0.7)
- Low-risk patients (predicted probability < 0.3)
- Correctly predicted vs. misclassified (if labels available)
- High-confidence vs. uncertain predictions

## Conditional Importance Aggregations

Context-dependent:
- For each feature, aggregate only patients where that feature value is extreme (top/bottom quartile)
- Aggregate interactions only when both features are simultaneously important
- Weight by local feature importance rank

## Top-K Aggregations

- For each feature, average SHAP values only from top K patients where it's most important
- Focuses on patients where feature truly matters

## Composite Metrics

Multi-dimensional aggregations:
- Magnitude × Consistency × Frequency
- Effect size / variability (signal-to-noise ratio)
- Impact score = mean(|SHAP|) × proportion_significant

## Temporal/Sequential (if data has temporal aspect)

- Early disease stage vs. late stage
- Pre/post intervention

## Matrix-Based Comparisons

Instead of single network:
- Differential networks - Compare high-risk vs. low-risk networks
- Consensus network - Only edges appearing in multiple subgroup networks
- Union network - All edges from any subgroup, colored by which groups

## Most Promising for Disease Research:

1) Risk-stratified networks - Most clinically interpretable
2) Consensus aggregation - Most robust to noise
3) Dominant direction - Clearest mechanistic interpretation
4) Clustering-based - May discover disease subtypes
5) Median + IQR - Robust summary with uncertainty quantification