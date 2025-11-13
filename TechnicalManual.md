# ADHD Digital Phenotype Analysis Pipeline - Complete Documentation

## 📊 Current Pipeline Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 0: DATA LOADING                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load Fitbit Data (fitbit_data.xlsx)                        │
│    - Read ONLY specified sheets: NC_Controls, ADHD_Unmedicated,│
│      ADHD_Stimulants                                            │
│    - Assign analysis_group (0, 1, 2) based on sheet name      │
│    - Combine sheets into single DataFrame                       │
│                                                                 │
│ 2. Load Metadata (ABCD_metadata_features_07232022.csv)        │
│    - Extract covariates: sex, interview_age                    │
│    - Keep only unique subjects                                  │
│                                                                 │
│ 3. Merge Datasets                                              │
│    - Left join Fitbit ← Metadata on 'subjectkey'              │
│    - Check covariate coverage (currently 52.5%)                │
│                                                                 │
│ 4. Prepare Covariates                                          │
│    - Convert categorical → dummy variables (sex → sex_M)       │
│    - Create final covariate column list                        │
│                                                                 │
│ 5. **FIX**: Convert Features to Numeric                        │
│    - pd.to_numeric() all 125 Fitbit features                   │
│    - Coerce errors to NaN                                      │
│                                                                 │
│ 6. Handle Missing Values                                       │
│    - Drop rows with missing features OR covariates             │
│    - Result: 2,491 → 1,309 subjects (52.5% retained)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1: RESIDUALIZATION (Covariate Adjustment)    │
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
│         PHASE 1.5: UNIVARIATE STATISTICAL TESTS                │
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
│ - univariate_tests_[comparison].csv (125 rows × 20 cols)     │
│ - effect_size_summary.csv (significant features only)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: PREDICTIVE MODELING                       │
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
│    - Confusion matrix                                          │
│    - ROC curves                                                │
│    - Calibration analysis                                      │
│                                                                 │
│ Outputs:                                                        │
│ - model_[comparison]_results.png (8-panel visualization)      │
│ - lr_coefficients_[comparison].csv                            │
│ - rf_importances_[comparison].csv                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 3: PCA VISUALIZATION                             │
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
│ - pca_2d_plot.png                                             │
│ - pca_3d_plot.png                                             │
│ - pca_scree_plot.png                                          │
│ - pca_loadings.png                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              FINAL: REPORTS & SUMMARY                          │
├─────────────────────────────────────────────────────────────────┤
│ - summary_report.txt (human-readable)                         │
│ - pipeline_results.json (machine-readable)                    │
│ - pipeline.log (detailed execution trace)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
project/
├── config.py                      # All parameters & paths
├── utils.py                       # Helper functions
├── data_loader.py                 # Phase 0
├── residualization.py             # Phase 1
├── univariate_tests.py            # Phase 1.5
├── predictive_models.py           # Phase 2
├── visualization.py               # Phase 3
├── main_pipeline.py               # Orchestrator
├── requirements.txt               # Dependencies
├── add_covariates_helper.py       # Optional utility
└── analysis_output/               # All results
    ├── residualized_data.csv
    ├── residualization_statistics.csv
    ├── univariate_tests_*.csv
    ├── effect_size_summary.csv
    ├── lr_coefficients_*.csv
    ├── rf_importances_*.csv
    ├── model_*_results.png
    ├── pca_*.png
    ├── summary_report.txt
    ├── pipeline_results.json
    └── pipeline.log
```

## 🔄 Data Flow

```
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
```

## 🎯 Current Sample Sizes

| Group | Raw | After Covariate Merge | After Cleaning |
|-------|-----|----------------------|----------------|
| **NC Controls** | 2,238 | 2,238 | **1,171** (52.3%) |
| **ADHD Unmedicated** | 143 | 143 | **74** (51.7%) |
| **ADHD Stimulants** | 110 | 110 | **64** (58.2%) |
| **TOTAL** | **2,491** | 2,491 | **1,309** (52.5%) |

## ⚠️ Critical Issue Identified

### Problem: Data Type Mismatch
```python
# Features stored as 'object' dtype (strings/mixed)
df['avg_hr_deep_mean'].dtype
>>> dtype('O')  # Should be float64

# Causes OLS regression to fail:
"Pandas data cast to numpy dtype of object"
```

### Root Cause:
Excel import with mixed content (numbers + text like '#N/A', 'NULL', empty strings)

### Solution Applied:
```python
# In get_available_features():
for feature in available_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    # Converts: '123.4' → 123.4, '#N/A' → NaN, 'NULL' → NaN
```

## 📊 Expected Results After Fix

**Phase 1 (Residualization):**
```
Successful: 125 features (was: 0)
Failed: 0 features (was: 125)
R² range: 0.05 - 0.35 (was: N/A)
```

**Phase 1.5 (Univariate Tests):**
```
Valid comparisons: ~375 (was: 0)
Significant features (q<0.05): ~20-40 per comparison
```

**Phase 2 (Models):**
```
Features available: 125 (was: 0)
Models trained: 6 (was: 0)
Expected AUC: 0.55-0.75 based on previous run
```

**Phase 3 (PCA):**
```
PC1 variance: ~16%
PC2 variance: ~13%
Cumulative (PC1-10): ~70%
```

## 🔧 Key Configuration Parameters

### Covariates (config.py)
```python
COVARIATES = ['sex', 'interview_age']  
# Missing: family_income, parent_grade, parent_div_cat, adopted
# Coverage: 52.5% (1,309/2,491 subjects)
```

### Model Parameters
```python
# Elastic Net Logistic Regression
penalty='elasticnet', l1_ratio=0.5, C=1.0, class_weight='balanced'

# Random Forest
max_depth=5, min_samples_leaf=5, max_features='sqrt', oob_score=True
```

### Comparison Pairs
```python
1. Control (n=1,171) vs Unmedicated (n=74)
2. Control (n=1,171) vs Stimulant (n=64)
3. Unmedicated (n=74) vs Stimulant (n=64)
```

## 🚀 Running the Pipeline

```bash
# Standard run (after fix)
python main_pipeline.py

# With hyperparameter tuning (slower, better results)
python main_pipeline.py --tune

# Skip validation (faster)
python main_pipeline.py --skip-validation
```

## 📈 Interpretation Guide

### Residualization R²
- **R² = 0.0**: Covariate has no effect (or feature is categorical)
- **R² = 0.1-0.3**: Moderate demographic influence (expected)
- **R² > 0.4**: Strong age/sex effect (e.g., resting heart rate)

### Effect Sizes (Cohen's d)
- **|d| < 0.2**: Negligible
- **|d| = 0.2-0.5**: Small
- **|d| = 0.5-0.8**: Medium
- **|d| > 0.8**: Large

### Model Performance
- **AUC = 0.50**: No better than chance
- **AUC = 0.70**: Acceptable discrimination
- **AUC = 0.80**: Excellent discrimination
- **AUC = 0.90+**: Outstanding (rare in behavioral data)

### PCA Interpretation
- **Clear separation**: Groups have distinct phenotypes
- **Overlap**: Shared physiology, subtle differences
- **No separation**: Differences are multivariate, not in dominant axes

## 🔄 Next Steps After Fix

1. **Run pipeline** → Check residualization R² values
2. **Review univariate results** → Which features differ significantly?
3. **Examine model performance** → AUCs, feature importances
4. **Interpret PCA** → Do groups separate visually?
5. **Add more covariates** → Merge family_income, parent_grade if available

---

**Version:** 2.0 (with data type fix)  
**Last Updated:** November 12, 2025  
**Status:** Ready to run after numeric conversion fix