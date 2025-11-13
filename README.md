ADHD Digital Phenotype Analysis Pipeline - Technical Manual

(Version 2.0 - Based on sheet-based groups and metadata file)

📊 Current Pipeline Structure

This document describes the exact flow orchestrated by main_pipeline.py.

┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 0: DATA LOADING                        │
│                  (data_loader.py, config.py)                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load Fitbit Data (config.FITBIT_FILE)                      │
│    - Read ONLY specified sheets (config.SHEETS_TO_LOAD):      │
│      'NC_Controls', 'ADHD_Unmedicated', 'ADHD_Stimulants'       │
│    - Assign analysis_group (0, 1, 2) based on sheet name      │
│    - Combine sheets into single DataFrame                       │
│                                                                 │
│ 2. Load Metadata (config.METADATA_FILE)                       │
│    - Extract covariates (config.COVARIATES): 'sex', 'interview_age'│
│    - Keep only unique subjects                                  │
│                                                                 │
│ 3. Merge Datasets                                              │
│    - Left join Fitbit ← Metadata on 'subjectkey'              │
│    - Check covariate coverage (logs show ~52.5%)               │
│                                                                 │
│ 4. Prepare Covariates                                          │
│    - Convert categorical → dummy variables ('sex' → 'sex_M')   │
│    - Create final covariate column list                        │
│                                                                 │
│ 5. Convert Features to Numeric (CRITICAL FIX)                  │
│    - pd.to_numeric() all 125 Fitbit features                   │
│    - Coerce errors ('#N/A', 'NULL') to NaN                     │
│                                                                 │
│ 6. Handle Missing Values (utils.py)                            │
│    - Drop rows with missing features OR covariates             │
│    - Result: 2,491 → 1,309 subjects (52.5% retained)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        PHASE 1: RESIDUALIZATION (Covariate Adjustment)          │
│                   (residualization.py)                          │
├─────────────────────────────────────────────────────────────────┤
│ For each of 125 Fitbit features:                              │
│                                                                 │
│ 1. Isolate Healthy Controls (n=1,171)                         │
│                                                                 │
│ 2. Fit OLS Regression (controls only):                        │
│    feature ~ sex_M + interview_age + intercept                │
│                                                                 │
│ 3. Predict for ALL subjects (n=1,309)                         │
│                                                                 │
│ 4. Calculate Residuals:                                       │
│    residual = actual_value - predicted_value                   │
│                                                                 │
│ 5. Store residualized feature (demographic effects removed)    │
│                                                                 │
│ Outputs:                                                        │
│ - residualized_data.csv (1,309 × 127 columns)                 │
│ - residualization_statistics.csv (R², p-values per feature)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 1.5: UNIVARIATE STATISTICAL TESTS                 │
│                 (univariate_tests.py)                           │
├─────────────────────────────────────────────────────────────────┤
│ For each feature × comparison pair:                            │
│                                                                 │
│ 1. Extract groups (e.g., Controls vs Unmedicated)             │
│                                                                 │
│ 2. Calculate Statistics:                                       │
│    - Group means, SDs, medians                                 │
│    - Independent t-test → t-stat, p-value                     │
│    - Mann-Whitney U test → U-stat, p-value                    │
│    - Cohen's d effect size                                     │
│                                                                 │
│ 3. FDR Correction (Benjamini-Hochberg):                       │
│    - Convert p-values → q-values                              │
│    - Flag significant features (q < 0.05)                     │
│                                                                 │
│ Outputs:                                                        │
│ - univariate_tests_[comparison].csv (125 rows × ~20 cols)     │
│ - effect_size_summary.csv (significant features only)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: PREDICTIVE MODELING                       │
│                (predictive_models.py)                           │
├─────────────────────────────────────────────────────────────────┤
│ For each comparison pair (3 comparisons):                      │
│                                                                 │
│ 1. Prepare Binary Classification Data:                        │
│    - Filter to 2 groups                                        │
│    - Create binary labels (0/1)                               │
│    - Remove zero-variance features                            │
│                                                                 │
│ 2. Train/Test Split (70/30, stratified):                      │
│    - Ensures balanced class representation                     │
│                                                                 │
│ 3. Model A: Elastic Net Logistic Regression                   │
│    - L1 + L2 regularization (l1_ratio=0.5)                   │
│    - Class weights balanced (handles 1,171 vs 74 vs 64)      │
│    - 5-fold cross-validation                                   │
│    - Output: Coefficients (feature importance)                 │
│                                                                 │
│ 4. Model B: Regularized Random Forest                         │
│    - max_depth=5, min_samples_leaf=5 (prevent overfitting)   │
│    - max_features='sqrt' (~11 features per split)             │
│    - Out-of-bag error tracking                                 │
│    - Output: Feature importances                               │
│                                                                 │
│ 5. Evaluation Metrics:                                        │
│    - Accuracy, ROC-AUC, Precision, Recall, F1                 │
│    - Confusion matrix, ROC curves, Calibration analysis       │
│                                                                 │
│ Outputs:                                                        │
│ - model_[comparison]_results.png (8-panel visualization)      │
│ - lr_coefficients_[comparison].csv                            │
│ - rf_importances_[comparison].csv                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 3: PCA VISUALIZATION                             │
│                  (visualization.py)                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Standardize Features (StandardScaler)                      │
│                                                                 │
│ 2. Fit PCA (10 components for scree, 2-3 for visualization)   │
│                                                                 │
│ 3. Statistical Testing:                                        │
│    - ANOVA on PC1, PC2 by group                               │
│    - Test if groups separate in PCA space                     │
│                                                                 │
│ 4. Visualizations:                                            │
│    - Scree plot (explained variance)                          │
│    - 2D scatter (PC1 vs PC2, colored by group)               │
│    - 3D scatter (PC1 vs PC2 vs PC3)                          │
│    - Loading heatmap (top 30 features)                        │
│                                                                 │
│ Outputs:                                                        │
│ - pca_2d_plot.png, pca_3d_plot.png, pca_scree_plot.png        │
│ - pca_loadings.png                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              FINAL: REPORTS & SUMMARY                          │
│                      (utils.py)                                 │
├─────────────────────────────────────────────────────────────────┤
│ - summary_report.txt (human-readable)                         │
│ - pipeline_results.json (machine-readable)                    │
│ - pipeline.log (detailed execution trace)                     │
└─────────────────────────────────────────────────────────────────┘


🔄 Data Flow & Sample Sizes

This flow traces the number of subjects through the pipeline.

Group

Raw (From Excel Sheets)

After Covariate Merge

After Cleaning (dropna)

NC Controls

2,238

2,238

1,171 (52.3%)

ADHD Unmedicated

143

143

74 (51.7%)

ADHD Stimulants

110

110

64 (58.2%)

TOTAL

2,491

2,491

1,309 (52.5%)

Data Flow Summary:
fitbit_data.xlsx (3 sheets)
↓ [load_fitbit_data()]
2,491 rows × ~130 columns
↓ [merge with metadata]
2,491 rows × ~136 columns (added sex, age)
↓ [convert to numeric + handle missing]
1,309 rows × 136 columns (52.5% retained)
↓ [residualize]
1,309 rows × 127 columns (residualized features)
↓ [univariate tests]
375 statistical comparisons (125 features × 3 pairs)
↓ [train models]
6 models (2 algorithms × 3 comparisons)
↓ [PCA]
1,309 points in 10D → 2D/3D projections

⚠️ Critical Issue Identified (and Fixed)

Problem: Data Type Mismatch

Symptom: OLS regression in Phase 1 fails with "Pandas data cast to numpy dtype of object".

Root Cause: Excel import with mixed content (numbers + text like '#N/A', 'NULL', empty strings) causes feature columns to be stored as dtype('O') (object/string) instead of float64.

Solution Applied (in data_loader.py): A loop was added in get_available_features() to force-convert all feature columns to numeric type before any analysis.

# In data_loader.py -> get_available_features():
for feature in available_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    # Converts: '123.4' → 123.4, '#N/A' → NaN, 'NULL' → NaN


This fix is essential and allows the pipeline to run end-to-end.

🔧 Key Configuration Parameters (config.py)

Covariates

# Primary covariates used for residualization
COVARIATES = ['sex', 'interview_age']  
# Note: family_income, parent_grade, etc. are NOT used in this config
# Coverage: 52.5% (1,309/2,491 subjects)


Model Parameters (Tuned for small N)

# Elastic Net Logistic Regression (handles correlated features)
penalty='elasticnet', l1_ratio=0.5, C=1.0, class_weight='balanced'

# Random Forest (conservative to prevent overfitting)
max_depth=5, min_samples_leaf=5, max_features='sqrt', oob_score=True


Comparison Pairs

Control (n=1,171) vs Unmedicated (n=74)

Control (n=1,171) vs Stimulant (n=64)

Unmedicated (n=74) vs Stimulant (n=64)

📈 Interpretation Guide

Residualization R² (residualization_statistics.csv)

R² = 0.0: Covariate has no effect (or feature is categorical).

R² = 0.1-0.3: Moderate demographic influence (expected).

R² > 0.4: Strong age/sex effect (e.g., resting heart rate).

Effect Sizes (Cohen's d) (univariate_tests_...csv)

|d| < 0.2: Negligible

|d| = 0.2-0.5: Small

|d| = 0.5-0.8: Medium

|d| > 0.8: Large

Model Performance (model_..._results.png)

AUC = 0.50: No better than chance.

AUC = 0.70: Acceptable discrimination.

AUC = 0.80: Excellent discrimination.

AUC = 0.90+: Outstanding (rare in behavioral data).

PCA Interpretation (pca_2d_plot.png)

Clear separation: Groups have distinct phenotypes.

Overlap: Shared physiology, subtle differences.

No separation: Differences are multivariate, not in dominant axes.