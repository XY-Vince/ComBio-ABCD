"""
Data Loading and Preprocessing Module
Handles loading Fitbit data from multiple sheets, metadata, and initial data preparation
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
import config
import utils

logger = logging.getLogger(__name__)


def load_metadata(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load ABCD metadata with covariates
    
    Parameters:
    -----------
    filepath : str, optional
        Path to metadata file. If None, uses config.METADATA_FILE
        
    Returns:
    --------
    pd.DataFrame : Loaded metadata
    """
    if filepath is None:
        filepath = config.METADATA_FILE
    
    logger.info(f"Loading metadata from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(df)} metadata records")
        
        # Check which covariates are available
        available_covs = [c for c in config.COVARIATES if c in df.columns]
        missing_covs = [c for c in config.COVARIATES if c not in df.columns]
        
        logger.info(f"Available covariates: {available_covs}")
        if missing_covs:
            logger.warning(f"Missing covariates: {missing_covs}")
        
        # Keep only relevant columns
        keep_cols = ['subjectkey'] + available_covs
        df_subset = df[keep_cols].copy()
        
        # Remove duplicates based on subjectkey
        df_subset = df_subset.drop_duplicates(subset=['subjectkey'])
        
        logger.info(f"Metadata prepared: {len(df_subset)} unique subjects with {len(available_covs)} covariates")
        
        return df_subset
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise


def load_fitbit_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load Fitbit data from Excel file with multiple sheets
    Only loads sheets specified in config.SHEETS_TO_LOAD
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the Fitbit data file. If None, uses config.FITBIT_FILE
        
    Returns:
    --------
    pd.DataFrame : Combined Fitbit data from specified sheets with group labels
    """
    if filepath is None:
        filepath = config.FITBIT_FILE
    
    logger.info(f"Loading Fitbit data from {filepath}")
    
    try:
        # Read Excel file and get all sheet names
        excel_file = pd.ExcelFile(filepath)
        all_sheet_names = excel_file.sheet_names
        
        logger.info(f"Found {len(all_sheet_names)} sheets in file: {all_sheet_names}")
        
        # Determine which sheets to load
        if config.SHEETS_TO_LOAD is not None:
            sheets_to_process = config.SHEETS_TO_LOAD
            logger.info(f"Loading ONLY specified sheets: {sheets_to_process}")
            
            # Verify specified sheets exist
            missing_sheets = [s for s in sheets_to_process if s not in all_sheet_names]
            if missing_sheets:
                logger.error(f"Specified sheets not found in file: {missing_sheets}")
                raise ValueError(f"Sheets not found: {missing_sheets}")
        else:
            sheets_to_process = all_sheet_names
            logger.info(f"Loading ALL sheets")
        
        # Load each specified sheet
        all_dfs = []
        
        for sheet_name in sheets_to_process:
            logger.info(f"Loading sheet: {sheet_name}")
            df_sheet = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Add a column to identify which sheet this came from
            df_sheet['source_sheet'] = sheet_name
            
            # Assign analysis_group based on exact sheet name mapping
            if sheet_name in config.SHEET_TO_GROUP:
                group_id = config.SHEET_TO_GROUP[sheet_name]
                df_sheet['analysis_group'] = group_id
                group_name = config.GROUP_LABELS[group_id]
                logger.info(f"  ✓ Assigned to {group_name}: {len(df_sheet)} subjects")
            else:
                logger.error(f"  ✗ Sheet '{sheet_name}' not in SHEET_TO_GROUP mapping!")
                raise ValueError(f"Unknown sheet: {sheet_name}. Update config.SHEET_TO_GROUP")
            
            all_dfs.append(df_sheet)
        
        # Combine all sheets
        df_combined = pd.concat(all_dfs, ignore_index=True)
        
        logger.info(f"Successfully loaded {len(df_combined)} total rows from {len(sheets_to_process)} sheets")
        
        # Check if subject key column exists
        if 'subjectkey' not in df_combined.columns:
            logger.error("'subjectkey' column not found in data")
            raise ValueError("'subjectkey' column is required")
        
        # Log group distribution
        group_counts = df_combined['analysis_group'].value_counts().sort_index()
        logger.info("=" * 60)
        logger.info("FINAL GROUP DISTRIBUTION:")
        logger.info("=" * 60)
        for group_id, count in group_counts.items():
            group_name = config.GROUP_LABELS.get(group_id, f"Group {group_id}")
            logger.info(f"  {group_name}: {count} subjects")
        logger.info("=" * 60)
        
        return df_combined
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading Fitbit data: {str(e)}")
        raise


def merge_datasets(fitbit_df: pd.DataFrame,
                  metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge Fitbit data with metadata covariates
    
    Parameters:
    -----------
    fitbit_df : pd.DataFrame
        Fitbit data with analysis_group already assigned
    metadata_df : pd.DataFrame, optional
        Metadata with covariates
        
    Returns:
    --------
    pd.DataFrame : Merged dataset
    """
    logger.info("Preparing merged dataset")
    
    df_merged = fitbit_df.copy()
    
    # Merge with metadata covariates
    if metadata_df is not None:
        logger.info("Merging with metadata covariates...")
        n_before = len(df_merged)
        # Ensure subjectkey is of the same type if possible, or handle as object
        if 'subjectkey' in metadata_df.columns:
            try:
                # Try to force string type for robust merging
                df_merged['subjectkey'] = df_merged['subjectkey'].astype(str)
                metadata_df['subjectkey'] = metadata_df['subjectkey'].astype(str)
            except Exception as e:
                logger.warning(f"Could not force string type on 'subjectkey' for merging: {e}")

            df_merged = df_merged.merge(metadata_df, on='subjectkey', how='left', suffixes=('', '_meta'))
            n_after = len(df_merged)
            
            if n_before != n_after:
                logger.warning(f"Row count changed during metadata merge: {n_before} -> {n_after}. This might indicate duplicate subjectkeys.")
                # Deduplicate if merge created extras, keeping the first
                df_merged = df_merged.drop_duplicates(subset=['subjectkey'], keep='first')
                logger.info(f"DataFrame deduplicated. New count: {len(df_merged)}")

            # Check covariate coverage
            available_covs = [c for c in config.COVARIATES if c in df_merged.columns]
            if not available_covs:
                 logger.warning("No covariates from config file found in merged data.")
            
            for cov in available_covs:
                n_available = df_merged[cov].notna().sum()
                pct_available = (n_available / len(df_merged)) * 100
                logger.info(f"  {cov}: {n_available}/{len(df_merged)} ({pct_available:.1f}%) available")
                
                if pct_available < 50:
                    logger.warning(f"    Low coverage for {cov}: only {pct_available:.1f}%")
        else:
            logger.warning("'subjectkey' not found in metadata_df. Cannot merge covariates.")
    else:
        logger.warning("No metadata (covariates) file provided or loaded. Proceeding without covariates.")
    
    # Log final group distribution
    group_counts = df_merged['analysis_group'].value_counts().sort_index()
    logger.info("Final analysis group distribution:")
    for group_id, count in group_counts.items():
        group_name = config.GROUP_LABELS.get(group_id, f"Group {group_id}")
        logger.info(f"  {group_name}: {count}")
    
    return df_merged


def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare covariate columns for regression
    Convert categorical variables to dummy variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : DataFrame with prepared covariates
    """
    logger.info("Preparing covariates")
    
    df_prep = df.copy()
    
    # Check which covariates exist in the dataframe
    missing_covariates = [cov for cov in config.COVARIATES if cov not in df.columns]
    if missing_covariates:
        logger.warning(f"Missing covariates listed in config: {missing_covariates}")
        logger.warning("Residualization will proceed with available covariates only")
    
    # Convert categorical covariates to dummy variables
    for cat_var in config.CATEGORICAL_COVARIATES:
        if cat_var in df_prep.columns:
            if df_prep[cat_var].isna().all():
                logger.warning(f"Categorical covariate '{cat_var}' is all NaN. Skipping dummy encoding.")
                continue
            
            logger.info(f"Converting {cat_var} to dummy variables")
            try:
                # (FIX: Set dummy_na=False. Missing rows are dropped by handle_missing_values)
                dummies = pd.get_dummies(df_prep[cat_var], prefix=cat_var, drop_first=True, dummy_na=False)
                df_prep = pd.concat([df_prep, dummies], axis=1)
                # Keep original column for reference but use dummies in analysis
            except Exception as e:
                logger.error(f"Failed to create dummy variables for {cat_var}: {e}")
    
    return df_prep


def get_available_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of Fitbit features that are actually present in the dataframe
    Also ensures features are numeric type
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (MODIFIED IN-PLACE)
        
    Returns:
    --------
    list : List of available feature names
    """
    available_features = [f for f in config.ALL_FITBIT_FEATURES if f in df.columns]
    
    missing_features = set(config.ALL_FITBIT_FEATURES) - set(available_features)
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features from config list")
        logger.debug(f"Missing features: {list(missing_features)[:10]}... (showing first 10)")
    
    logger.info(f"Found {len(available_features)} available Fitbit features from config list")
    
    # CRITICAL FIX: Convert all features to numeric, handling errors
    logger.info("Converting features to numeric type...")
    conversion_issues = []
    
    # We must operate directly on the DataFrame 'df' for changes to stick
    for feature in available_features:
        try:
            # Convert to numeric, coercing errors to NaN
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        except Exception as e:
            conversion_issues.append(feature)
            logger.warning(f"Could not convert {feature} to numeric: {e}")
            
    if conversion_issues:
        logger.warning(f"{len(conversion_issues)} features had conversion issues")
    
    logger.info("Feature type conversion complete")
    
    return available_features


def load_and_prepare_data(fitbit_path: Optional[str] = None,
                          metadata_path: Optional[str] = None) -> Tuple[pd.DataFrame, list]:
    """
    Main function to load and prepare all data
    
    Parameters:
    -----------
    fitbit_path : str, optional
        Path to Fitbit data
    metadata_path : str, optional
        Path to metadata with covariates
        
    Returns:
    --------
    Tuple[pd.DataFrame, list] : Prepared dataframe and list of available features
    """
    logger.info("=" * 80)
    logger.info("Starting data loading and preparation")
    logger.info("=" * 80)
    
    # Load Fitbit data (handles multiple sheets automatically)
    fitbit_df = load_fitbit_data(fitbit_path)
    
    # Load metadata with covariates
    metadata_df = None
    if metadata_path or config.METADATA_FILE:
        try:
            metadata_df = load_metadata(metadata_path)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            logger.warning("Proceeding without metadata covariates")
    
    # Merge all datasets
    df_merged = merge_datasets(fitbit_df, metadata_df)
    
    # --- START OF BUG FIX ---
    # We must drop rows *before* converting to numeric
    
    # Get list of all columns to check for missingness
    # Use the config list, as 'get_available_features' hasn't run yet
    all_check_cols = config.ALL_FITBIT_FEATURES + config.COVARIATES
    
    # Handle missing values
    df_clean = utils.handle_missing_values(
        df_merged,
        all_check_cols,
        method=config.MISSING_VALUE_METHOD
    )
    
    # NOW, on the cleaned (row-dropped) dataframe, find features and convert to numeric
    # This modifies 'df_clean' in-place and returns the list of feature names
    available_features = get_available_features(df_clean)
    
    # --- END OF BUG FIX ---
    
    # Perform initial data quality check
    quality_report = utils.check_data_quality(
        df_clean, 
        available_features, 
        'analysis_group'
    )
    
    logger.info("=" * 80)
    logger.info(f"Data preparation complete. Final dataset: {len(df_clean)} subjects")
    logger.info("=" * 80)
    
    # Return the cleaned dataframe and the list of features that were found
    return df_clean, available_features


if __name__ == "__main__":
    logger.info("This script is a module and is intended to be run by main_pipeline.py")
    pass