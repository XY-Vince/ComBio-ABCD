"""
THE REAL CULPRIT: Boolean dtype covariate (sex_M)

When statsmodels OLS sees a DataFrame with mixed dtypes (float64 + bool),
it converts everything to object dtype, causing the error.

SOLUTION: Force ALL covariates to float64, including boolean dummies

This is a simple one-line change in residualize_feature()
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Tuple, Dict
from loguru import logger

# ============================================================================
# THE FIX: In residualize_feature(), change these lines:
# ============================================================================

# CURRENT CODE (lines ~91-94):
"""
# FIX: Force covariates to numeric
for col in X_controls.columns:
    if col in covariate_cols:
        X_controls[col] = pd.to_numeric(X_controls[col], errors='coerce')
X_controls = X_controls.dropna()
"""

# SHOULD BE:
"""
# FIX: Force covariates to numeric (including booleans!)
for col in X_controls.columns:
    X_controls[col] = pd.to_numeric(X_controls[col], errors='coerce').astype('float64')
X_controls = X_controls.dropna()
"""

# SAME FIX needed in two other places:

# CURRENT CODE (lines ~118-121):
"""
# FIX: Force covariates to numeric for all subjects
for col in X_all.columns:
    if col in covariate_cols:
        X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
"""

# SHOULD BE:
"""
# FIX: Force covariates to numeric for all subjects (including booleans!)
for col in X_all.columns:
    X_all[col] = pd.to_numeric(X_all[col], errors='coerce').astype('float64')
"""


# ============================================================================
# COMPLETE CORRECTED residualize_feature() FUNCTION
# Replace the entire function with this:
# ============================================================================

def residualize_feature(feature_name: str,
                       df_controls: pd.DataFrame,
                       df_all: pd.DataFrame,
                       covariate_cols: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    Residualize a single feature using control group regression
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature to residualize
    df_controls : pd.DataFrame
        Data for healthy controls only
    df_all : pd.DataFrame
        Data for all subjects
    covariate_cols : List[str]
        List of covariate column names
        
    Returns:
    --------
    Tuple[np.ndarray, Dict] : Residualized values and model statistics
    """
    try:
        # Get feature data for controls
        y_controls = df_controls[feature_name].dropna()
        # FIX: Force to numeric after extraction
        y_controls = pd.to_numeric(y_controls, errors='coerce').dropna()
        
        if len(y_controls) < 10:
            logger.warning(f"Insufficient control data for {feature_name} (n={len(y_controls)})")
            return None, None
        
        # Get covariates for the same subjects
        X_controls = df_controls.loc[y_controls.index, covariate_cols].dropna()

        # FIX: Force ALL covariates to float64 (including booleans!)
        for col in X_controls.columns:
            X_controls[col] = pd.to_numeric(X_controls[col], errors='coerce').astype('float64')
        X_controls = X_controls.dropna()
        
        # Align y and X (keep only subjects with all data)
        common_idx = y_controls.index.intersection(X_controls.index)
        y_controls = y_controls.loc[common_idx]
        X_controls = X_controls.loc[common_idx]
        
        if len(y_controls) < len(covariate_cols) + 5:
            logger.warning(f"Insufficient data for {feature_name} after alignment (n={len(y_controls)})")
            return None, None
        
        # Add constant for intercept
        X_controls_const = sm.add_constant(X_controls, has_constant='add')
        
        # Fit OLS model on controls
        model = sm.OLS(y_controls, X_controls_const, missing='drop').fit()
        
        # Prepare covariates for all subjects
        X_all = df_all[covariate_cols].copy()

        # FIX: Force ALL covariates to float64 (including booleans!)
        for col in X_all.columns:
            X_all[col] = pd.to_numeric(X_all[col], errors='coerce').astype('float64')
        
        X_all_const = sm.add_constant(X_all, has_constant='add')
        
        # Ensure X_all has the same columns in the same order as X_controls
        X_all_const = X_all_const[X_controls_const.columns]
        
        # Predict for all subjects
        predicted = model.predict(X_all_const)
        
        # Calculate residuals
        # FIX: Force feature to numeric before subtraction
        feature_values = pd.to_numeric(df_all[feature_name], errors='coerce')
        residuals = feature_values - predicted
        
        # Store model statistics
        stats = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_pvalue': model.f_pvalue,
            'n_controls': len(y_controls),
            'coefficients': dict(zip(X_controls_const.columns, model.params))
        }
        
        return residuals.values, stats
        
    except (ValueError, KeyError, np.linalg.LinAlgError) as e:
        logger.error(f"Error residualizing {feature_name}: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"UNEXPECTED Error residualizing {feature_name}: {str(e)}", exc_info=True)
        return None, None


# ============================================================================
# KEY CHANGES FROM PREVIOUS VERSION:
# ============================================================================
# 1. Line ~92: Added .astype('float64') to convert booleans
#    X_controls[col] = pd.to_numeric(...).astype('float64')
#
# 2. Line ~119: Added .astype('float64') to convert booleans  
#    X_all[col] = pd.to_numeric(...).astype('float64')
#
# 3. Removed the "if col in covariate_cols" check since we're looping
#    over X_controls.columns/X_all.columns which already ARE the covariates