"""
Configuration file for ADHD Digital Phenotype Analysis Pipeline
Contains all parameters, file paths, and feature definitions
"""

import os
from pathlib import Path

# ============================================================================
# FILE PATHS (FIX 1: Relative Paths)
# ============================================================================
# Use relative paths based on this file's location
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
SOURCE_DIR = DATA_DIR / "source_files"
OUTPUT_DIR = DATA_DIR / "analysis_output"

# Define data files
FITBIT_FILE = SOURCE_DIR / "fitbit_data.xlsx"
METADATA_FILE = SOURCE_DIR / "ABCD_metadata_features_07232022.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SHEET SELECTION
# ============================================================================
# Specify EXACTLY which sheets to load (set to None to load all sheets)
SHEETS_TO_LOAD = ['NC_Controls', 'ADHD_Unmedicated', 'ADHD_Stimulants']

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5
VARIANCE_THRESHOLD = 0.01  # For feature selection

# (FIX 2: Make missing value method explicit)
MISSING_VALUE_METHOD = 'drop'  # Options: 'drop', 'median', 'mean'

# (FIX 3: Removed USE_MEDICATION_DATA - no longer needed)
# ============================================================================
# GROUP DEFINITIONS
# ============================================================================
# Define how clinical groups are encoded
GROUP_DEFINITIONS = {
    'control': 0,
    'unmedicated_adhd': 1,
    'stimulant_adhd': 2,
}

GROUP_LABELS = {
    0: 'Healthy Control',
    1: 'ADHD Unmedicated',
    2: 'ADHD Stimulant',
}

# Sheet name to group mapping (exact names)
SHEET_TO_GROUP = {
    'NC_Controls': 0,
    'ADHD_Unmedicated': 1,
    'ADHD_Stimulants': 2,
}

# ============================================================================
# COVARIATES (Confounders to control for)
# ============================================================================
# GROUP 1: BASIC COVARIATES (Demographics)
BASIC_COVARIATES = [
    'sex',              # Sex/Gender
    'interview_age'     # Age at interview
]

# GROUP 2: BEHAVIORAL COVARIATES (CBCL & DSM Scores)
BEHAVIORAL_COVARIATES = [
    'cbcl_internalizing',   # CBCL Internalizing Score
    'cbcl_externalizing',   # CBCL Externalizing Score
    'dsm_internalizing',    # DSM Internalizing Score
    'dsm_externalizing'     # DSM Externalizing Score
]

# GROUP 3: FAMILY HISTORY OF PSYCHIATRIC ILLNESS
FAMILY_PSYCH_HISTORY = [
    'famhis_bip',          # Family history: Bipolar Disorder
    'famhis_schiz',        # Family history: Schizophrenia
    'famhis_antisocial',   # Family history: Antisocial Behavior
    'famhis_nerves',       # Family history: Nervous Breakdown
    'famhis_treatment',    # Family history: Psychiatric Treatment
    'famhis_hospital',     # Family history: Hospital Admission
    'famhis_suicide'       # Family history: Suicide
]

# GROUP 4: FAMILY SITUATION
FAMILY_SITUATION = [
    'parent_div_cat',      # Divorced Parents (categorical)
    'parent_grade',        # Parents' Level of Education
    'family_income',       # Family Income
    'adopted'              # Adoption status
]

# Primary covariates to use in main analysis
COVARIATES = BASIC_COVARIATES + BEHAVIORAL_COVARIATES + FAMILY_PSYCH_HISTORY + FAMILY_SITUATION

# Categorical covariates that need dummy encoding
CATEGORICAL_COVARIATES = ['sex', 'parent_div_cat', 'adopted']

COVARIATE_GROUPS = {
    'basic': BASIC_COVARIATES,
    'behavioral': BEHAVIORAL_COVARIATES,
    'family_psych': FAMILY_PSYCH_HISTORY,
    'family_situation': FAMILY_SITUATION
}

# ============================================================================
# FITBIT FEATURES (FIX 4: Dynamic Feature Loading)
# ============================================================================
# This list is now used to *identify* features if a dynamic approach isn't
# preferred. For a fully dynamic approach, data_loader.py can be set
# to infer features automatically. We will keep the fix in data_loader.py
# that *uses* this list, as it's safer.
# Sleep features
SLEEP_FEATURES = [
    'avg_hr_deep_mean', 'avg_hr_rem_mean', 'first_inbed_minutes_min',
    'first_sleep_minutes_min', 'outbed_minutes_min', 'wakeup_minutes_min',
    'sleepperiod_minutes_min', 'wake_minutes_min', 'light_minutes_min',
    'deep_minutes_min', 'rem_minutes_min', 'wake_count_min',
    'avg_hr_wake_min', 'avg_hr_light_min', 'avg_hr_deep_min', 'avg_hr_rem_min',
    'first_inbed_minutes_max', 'first_sleep_minutes_max', 'outbed_minutes_max',
    'wakeup_minutes_max', 'sleepperiod_minutes_max', 'wake_minutes_max',
    'light_minutes_max', 'deep_minutes_max', 'rem_minutes_max', 'wake_count_max',
    'avg_hr_wake_max', 'avg_hr_light_max', 'avg_hr_deep_max', 'avg_hr_rem_max',
    'first_inbed_minutes_median', 'first_sleep_minutes_median',
    'outbed_minutes_median', 'wakeup_minutes_median', 'sleepperiod_minutes_median',
    'wake_minutes_median', 'light_minutes_median', 'deep_minutes_median',
    'rem_minutes_median', 'wake_count_median', 'avg_hr_wake_median',
    'avg_hr_light_median', 'avg_hr_deep_median', 'avg_hr_rem_median',
    'first_inbed_minutes_sd', 'first_sleep_minutes_sd', 'outbed_minutes_sd',
    'wakeup_minutes_sd', 'sleepperiod_minutes_sd', 'wake_minutes_sd',
    'light_minutes_sd', 'deep_minutes_sd', 'rem_minutes_sd', 'wake_count_sd',
    'avg_hr_wake_sd', 'avg_hr_light_sd', 'avg_hr_deep_sd', 'avg_hr_rem_sd',
]

# Activity features
ACTIVITY_FEATURES = [
    'day_total_steps_no_el_mean', 'day_min_mean', 'night_min_mean',
    'sleep_min_mean', 'excl_day_min_mean', 'excl_day_min_hr50_mean',
    'excl_day_min_nohr_mean', 'excl_day_min_hr_rept_mean', 'excl_night_min_mean',
    'excl_night_min_hr50_mean', 'excl_night_min_nohr_mean',
    'excl_night_min_hr_rept_mean', 'excl_sleep_min_mean', 'excl_sleep_min_hr50_mean',
    'excl_sleep_min_nohr_mean', 'excl_sleep_min_rept_mean', 'total_min_mean',
    'total_step_mean', 'total_ave_met_mean', 'total_sedentary_min_mean',
    'total_light_active_min_mean', 'total_fairly_active_min_mean',
    'total_very_active_min_mean', 'fitbit_totalsteps_mean',
    'fitbit_sedentarymin_mean', 'fitbit_lightlyactivemin_mean',
    'fitbit_fairlyactivemin_mean', 'fitbit_veryactivemin_mean',
    'fitbit_restingheartrate_mean', 'dayt_total_steps_mean',
    'dayt_ave_met_value_mean', 'dayt_sedentary_min_mean',
    'dayt_light_active_min_mean', 'dayt_farily_active_min_mean',
    'dayt_very_active_min_mean',
]

# Variability features (min, max, median, sd)
VARIABILITY_FEATURES = [
    'day_total_steps_no_el_min', 'day_min_min', 'night_min_min',
    'sleep_min_min', 'day_total_steps_no_el_max', 'day_min_max',
    'night_min_max', 'sleep_min_max', 'day_total_steps_no_el_median',
    'day_min_median', 'night_min_median', 'sleep_min_median',
    'day_total_steps_no_el_sd', 'day_min_sd', 'night_min_sd', 'sleep_min_sd',
    'total_min_min', 'total_step_min', 'total_ave_met_min',
    'total_sedentary_min_min', 'total_min_max', 'total_step_max',
    'total_ave_met_max', 'total_sedentary_min_max', 'total_min_median',
    'total_step_median', 'total_ave_met_median', 'total_sedentary_min_median',
    'total_min_sd', 'total_step_sd', 'total_ave_met_sd',
    'total_sedentary_min_sd',
]

# Combine all features
ALL_FITBIT_FEATURES = SLEEP_FEATURES + ACTIVITY_FEATURES + VARIABILITY_FEATURES

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
LOGISTIC_REGRESSION_PARAMS = {
    'penalty': 'elasticnet',
    'C': 1.0,
    'solver': 'saga',
    'l1_ratio': 0.5,
    'max_iter': 2000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'oob_score': True
}

# Hyperparameter search space for GridSearchCV
LOGISTIC_GRID = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],
}

RF_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'max_features': ['sqrt', 'log2']
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
COLOR_PALETTE = 'Set2'

PCA_COMPONENTS = 2
PCA_N_COMPONENTS_SCREE = 10

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = OUTPUT_DIR / 'pipeline.log'  # Use Path object

# ============================================================================
# DATA QUALITY PARAMETERS
# ============================================================================
MIN_SUBJECTS_PER_GROUP = 10
MAX_MISSING_RATE = 0.3
OUTLIER_THRESHOLD = 3

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================
ALPHA_LEVEL = 0.05
MULTIPLE_COMPARISON_METHOD = 'fdr_bh'

# ============================================================================
# COMPARISON PAIRS
# ============================================================================
COMPARISON_PAIRS = [
    ('control', 'unmedicated_adhd'),
    ('control', 'stimulant_adhd'),
    ('unmedicated_adhd', 'stimulant_adhd'),
]

# ============================================================================
# (FIX 5: Removed dead code FEATURE_CATEGORIES)
# ============================================================================