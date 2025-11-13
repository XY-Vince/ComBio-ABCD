"""
Phase 1: Residualization Module
Removes confounding effects of covariates from Fitbit features
Uses only healthy controls to build the regression models
"""

import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from typing import List, Tuple, Dict
import config
import utils
from data_loader import prepare_covariates # Import prepare_covariates

logger = logging.getLogger(__name__)


def get_covariate_columns(df: pd.DataFrame) -> List[str]:
    """
    Get the list of covariate columns including dummy variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with prepared covariates
        
    Returns:
    --------
    List[str] : List of all covariate column names
    """
    covariate_cols = []
    
    if len(config.COVARIATES) == 0:
        logger.warning("No covariates specified in config. Residualization will only mean-center features.")
        return covariate_cols
    
    for cov in config.COVARIATES:
        if cov in config.CATEGORICAL_COVARIATES:
            # Get all dummy variable columns for this categorical variable
            dummy_cols = [col for col in df.columns if col.startswith(f"{cov}_")]
            covariate_cols.extend(dummy_cols)
        else:
            # Continuous variable
            if cov in df.columns:
                covariate_cols.append(cov)
            else:
                logger.warning(f"Covariate {cov} not found in dataframe")
    
    # Ensure no duplicates
    covariate_cols = sorted(list(set(covariate_cols)))
    
    logger.info(f"Using {len(covariate_cols)} covariate columns: {covariate_cols}")
    return covariate_cols

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
        
        # === FIX: Force to numeric after extraction ===
        y_controls = pd.to_numeric(y_controls, errors='coerce').dropna()
        # === END FIX ===
        
        if len(y_controls) < 10:
            logger.warning(f"Insufficient control data for {feature_name} (n={len(y_controls)})")
            return None, None
        
        # Get covariates for the same subjects
        X_controls = df_controls.loc[y_controls.index, covariate_cols].dropna()

        # FIX: Force covariates to numeric (including booleans!)
        for col in X_controls.columns:
            X_controls[col] = pd.to_numeric(X_controls[col], errors='coerce').astype('float64')
        X_controls = X_controls.dropna()
        
        # === FIX: Force covariates to numeric ===
        for col in X_controls.columns:
            if col in covariate_cols:
                X_controls[col] = pd.to_numeric(X_controls[col], errors='coerce')
        X_controls = X_controls.dropna()
        # === END FIX ===
        
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

        # FIX: Force covariates to numeric for all subjects
        # FIX: Force covariates to numeric for all subjects (including booleans!)
        for col in X_all.columns:
            X_all[col] = pd.to_numeric(X_all[col], errors='coerce').astype('float64')
        
        # === FIX: Force covariates to numeric for all subjects ===
        for col in X_all.columns:
            if col in covariate_cols:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
        # === END FIX ===
        
        X_all_const = sm.add_constant(X_all, has_constant='add')
        
        # Ensure X_all has the same columns in the same order as X_controls
        X_all_const = X_all_const[X_controls_const.columns]
        
        # Predict for all subjects
        predicted = model.predict(X_all_const)
        
        # Calculate residuals
        # === FIX: Force feature to numeric before subtraction ===
        feature_values = pd.to_numeric(df_all[feature_name], errors='coerce')
        residuals = feature_values - predicted
        # === END FIX ===
        
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



def residualize_all_features(df: pd.DataFrame,
                            features: List[str],
                            control_group: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """
    Residualize all Fitbit features using healthy control regression
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with all subjects (cleaned)
    features : List[str]
        List of features to residualize
    control_group : int
        Value indicating healthy control group in analysis_group column
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict] : DataFrame with residualized features and statistics
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: RESIDUALIZATION - Removing Covariate Effects")
    logger.info("=" * 80)
    
    logger.info("Preparing covariates for residualization...")
    df_prepared = prepare_covariates(df)
    
    # === CORRECTED BUG FIX ===
    # prepare_covariates() creates a copy that loses numeric dtypes
    # We need to FORCE features back to float using .astype()
    logger.info("Re-validating numeric types on prepared data...")
    
    conversion_success = 0
    conversion_failures = []
    
    for feature in features:
        if feature in df_prepared.columns:
            try:
                # First try pd.to_numeric
                df_prepared[feature] = pd.to_numeric(df_prepared[feature], errors='coerce')
                
                # Then FORCE to float64 (this is the key!)
                df_prepared[feature] = df_prepared[feature].astype('float64')
                
                conversion_success += 1
                
            except Exception as e:
                logger.error(f"Failed to convert {feature}: {e}")
                conversion_failures.append(feature)
    
    logger.info(f"Type conversion: {conversion_success} success, {len(conversion_failures)} failures")
    
    # Verify it worked (check first 3 features)
    logger.info("Verifying dtypes after conversion:")
    for feature in features[:3]:
        if feature in df_prepared.columns:
            dtype = df_prepared[feature].dtype
            is_numeric = pd.api.types.is_numeric_dtype(df_prepared[feature])
            logger.info(f"  {feature}: dtype={dtype}, is_numeric={is_numeric}")
            
            # If STILL not numeric, something is very wrong
            if not is_numeric:
                logger.error(f"  ❌ CRITICAL: {feature} is STILL not numeric!")
                logger.error(f"     Sample values: {df_prepared[feature].head(3).tolist()}")
    # === END CORRECTED BUG FIX ===

    # Separate controls from full dataset
    df_controls = df_prepared[df_prepared['analysis_group'] == control_group].copy()
    logger.info(f"Using {len(df_controls)} healthy controls for regression models")
    
    if len(df_controls) < 20:
        logger.error(f"Insufficient control subjects (n={len(df_controls)}). Need at least 20.")
        raise ValueError("Insufficient control subjects for residualization")
    
    # Get covariate columns (from the prepared dataframe)
    covariate_cols = get_covariate_columns(df_prepared)
    
    # Initialize output dataframe
    df_residualized = pd.DataFrame(index=df_prepared.index)
    
    # Copy subject identifiers and group labels
    df_residualized['subjectkey'] = df_prepared['subjectkey']
    df_residualized['analysis_group'] = df_prepared['analysis_group']
    
    # Dictionary to store model statistics
    model_stats = {}
    
    # Residualize each feature
    successful_features = []
    failed_features = []
    
    logger.info(f"Residualizing {len(features)} features...")
    
    for i, feature in enumerate(features, 1):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(features)} features processed")
        
        if feature not in df_prepared.columns:
            logger.warning(f"Feature {feature} not in dataframe")
            failed_features.append(feature)
            continue
        
        # Check if feature is numeric before passing
        if not pd.api.types.is_numeric_dtype(df_prepared[feature]):
             logger.error(f"Feature {feature} is still not numeric. Skipping.")
             failed_features.append(feature)
             continue

        residuals, stats = residualize_feature(
            feature, df_controls, df_prepared, covariate_cols
        )
        
        if residuals is not None:
            df_residualized[feature] = residuals
            model_stats[feature] = stats
            successful_features.append(feature)
        else:
            failed_features.append(feature)
    
    logger.info("=" * 80)
    logger.info(f"Residualization complete:")
    logger.info(f"  Successful: {len(successful_features)} features")
    logger.info(f"  Failed: {len(failed_features)} features")
    logger.info("=" * 80)
    
    if failed_features:
        logger.warning(f"Failed features: {failed_features[:10]}...")  # Show first 10
    
    # Save residualization statistics
    if model_stats:
        stats_df = pd.DataFrame(model_stats).T
        stats_file = 'residualization_statistics.csv'
        stats_df.to_csv(config.OUTPUT_DIR / stats_file)
        logger.info(f"Residualization statistics saved to {stats_file}")
    
    return df_residualized, model_stats


def validate_residualization(df_original: pd.DataFrame,
                            df_residualized: pd.DataFrame,
                            features: List[str],
                            covariate: str = 'interview_age') -> Dict:
    """
    Validate that residualization successfully removed covariate effects
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe before residualization
    df_residualized : pd.DataFrame
        Dataframe after residualization
    features : List[str]
        List of features to validate
    covariate : str
        Covariate to check correlation with
        
    Returns:
    --------
    Dict : Validation statistics
    """
    logger.info("Validating residualization effectiveness...")
    
    validation_results = {
        'original_correlations': [],
        'residualized_correlations': [],
        'features': []
    }
    
    if covariate not in df_original.columns:
        logger.warning(f"Covariate {covariate} not available for validation")
        return validation_results
    
    # Check first 20 successful features
    features_to_check = [f for f in features if f in df_residualized.columns][:20]
    
    if not features_to_check:
        logger.warning("No successful features to validate.")
        return validation_results

    for feature in features_to_check:
        if feature in df_original.columns and feature in df_residualized.columns:
            # Original correlation
            orig_corr = df_original[feature].corr(df_original[covariate])
            
            # Residualized correlation
            resid_corr = df_residualized[feature].corr(df_original[covariate])
            
            validation_results['original_correlations'].append(orig_corr)
            validation_results['residualized_correlations'].append(resid_corr)
            validation_results['features'].append(feature)
    
    # Calculate mean absolute correlations, ignoring NaNs
    with np.errstate(invalid='ignore'):
        mean_orig = np.nanmean(np.abs(validation_results['original_correlations']))
        mean_resid = np.nanmean(np.abs(validation_results['residualized_correlations']))
        reduction = (1 - mean_resid / mean_orig) * 100 if mean_orig > 0 else 0.0
    
    logger.info(f"Validation results:")
    logger.info(f"  Mean |correlation| with {covariate} before: {mean_orig:.4f}")
    logger.info(f"  Mean |correlation| with {covariate} after: {mean_resid:.4f}")
    logger.info(f"  Reduction: {reduction:.1f}%")
    
    return validation_results


if __name__ == "__main__":
    logger.info("This script is a module and is intended to be run by main_pipeline.py")
    pass