"""
Utility functions for the ADHD Digital Phenotype Analysis Pipeline
Contains helper functions for data quality checks, statistics, and reporting
"""

import pandas as pd
import numpy as np
import logging
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import config
import json

# (FIX: Removed logging.basicConfig - moved to main_pipeline.py)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Handles Numpy data types during JSON serialization."""
    def default(self, obj):
        # Handle NumPy integers
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # Handle NumPy floats
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        # Handle NumPy boolean
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle sklearn objects (not serializable)
        elif isinstance(obj, (LogisticRegression, RandomForestClassifier, StandardScaler)):
            return f"{type(obj).__name__} object (not serializable)"
        # Handle pandas objects
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super(NumpyEncoder, self).default(obj)


def check_data_quality(df: pd.DataFrame, 
                       features: List[str], 
                       group_col: str = 'analysis_group') -> Dict:
    """
    Perform comprehensive data quality checks
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        List of feature columns to check
    group_col : str
        Column containing group labels
        
    Returns:
    --------
    Dict : Dictionary containing quality metrics
    """
    logger.info("Performing data quality checks...")
    
    quality_report = {
        'total_subjects': len(df),
        'group_counts': df[group_col].value_counts().to_dict(),
        'missing_features': {},
        'high_missing_features': [],
        'outlier_counts': {},
        'feature_stats': {}
    }
    
    # Check group sample sizes
    for group, count in quality_report['group_counts'].items():
        if count < config.MIN_SUBJECTS_PER_GROUP:
            logger.warning(f"Group {group} has only {count} subjects (minimum: {config.MIN_SUBJECTS_PER_GROUP})")
    
    # Check missing values
    for feature in features:
        if feature in df.columns:
            missing_rate = df[feature].isna().sum() / len(df)
            quality_report['missing_features'][feature] = float(missing_rate)
            
            if missing_rate > config.MAX_MISSING_RATE:
                quality_report['high_missing_features'].append(feature)
                logger.warning(f"Feature {feature} has {missing_rate:.2%} missing values")
    
    # Check for outliers
    for feature in features:
        if feature in df.columns:
            feature_data = df[feature].dropna()
            if len(feature_data) > 0 and pd.api.types.is_numeric_dtype(feature_data):
                z_scores = np.abs(stats.zscore(feature_data))
                outlier_count = int((z_scores > config.OUTLIER_THRESHOLD).sum())
                quality_report['outlier_counts'][feature] = outlier_count
                
                # Store basic statistics
                quality_report['feature_stats'][feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'median': float(df[feature].median())
                }
            elif not pd.api.types.is_numeric_dtype(feature_data):
                logger.warning(f"Feature {feature} is not numeric, skipping outlier check.")
    
    logger.info(f"Data quality check complete. Total subjects: {quality_report['total_subjects']}")
    logger.info(f"Features with high missing rate: {len(quality_report['high_missing_features'])}")
    
    return quality_report


def handle_missing_values(df: pd.DataFrame, 
                         features_and_covariates: List[str], 
                         method: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features_and_covariates : List[str]
        List of all columns to check for missingness
    method : str
        Method to handle missing values ('drop', 'median', 'mean')
        
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    logger.info(f"Handling missing values using method: {method}")
    
    # Start with a copy
    df_clean = df.copy()
    
    # Only check columns that actually exist in the dataframe
    existing_cols = [col for col in features_and_covariates if col in df_clean.columns]
    
    if method == 'drop':
        # Drop rows with any missing values in features or covariates
        n_before = len(df_clean)
        df_clean = df_clean.dropna(subset=existing_cols)
        n_after = len(df_clean)
        logger.info(f"Dropped {n_before - n_after} rows with missing values. Remaining: {n_after}")
        
    elif method == 'median':
        for col in existing_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        logger.info("Filled missing numeric values with median")
        
    elif method == 'mean':
        for col in existing_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                mean_val = df_clean[col].mean()
                df_clean[col].fillna(mean_val, inplace=True)
        logger.info("Filled missing numeric values with mean")
    
    return df_clean


def detect_outliers(df: pd.DataFrame, 
                   features: List[str], 
                   method: str = 'zscore',
                   threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Detect and optionally remove outliers
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        Features to check for outliers
    method : str
        Method for outlier detection ('zscore', 'iqr')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series] : DataFrame and outlier mask
    """
    logger.info(f"Detecting outliers using {method} method with threshold {threshold}")
    
    df_clean = df.copy()
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    
    for feature in features:
        if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
            continue
            
        feature_data = df[feature].dropna()
        if len(feature_data) == 0:
            continue
            
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(feature_data))
            feature_outliers = pd.Series([False] * len(df), index=df.index)
            feature_outliers.loc[feature_data.index] = z_scores > threshold
            
        elif method == 'iqr':
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        
        outlier_mask = outlier_mask | feature_outliers
    
    logger.info(f"Detected {outlier_mask.sum()} rows with outliers")
    
    return df_clean, outlier_mask


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size (FIX: Consolidated function)
    
    Parameters:
    -----------
    group1, group2 : np.ndarray
        Arrays of values for two groups
        
    Returns:
    --------
    float : Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0 # Return 0 if no variance
    
    return (mean1 - mean2) / pooled_std


def perform_feature_comparison(df: pd.DataFrame,
                               features: List[str],
                               group_col: str,
                               group1: int,
                               group2: int) -> pd.DataFrame:
    """
    Perform statistical comparison between two groups for all features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        List of features to compare
    group_col : str
        Column containing group labels
    group1, group2 : int
        Group identifiers to compare
        
    Returns:
    --------
    pd.DataFrame : Results with t-statistics, p-values, and effect sizes
    """
    logger.info(f"Comparing features between groups {group1} and {group2}")
    
    results = []
    
    for feature in features:
        if feature not in df.columns:
            continue
            
        data1 = df[df[group_col] == group1][feature].dropna()
        data2 = df[df[group_col] == group2][feature].dropna()
        
        if len(data1) < 3 or len(data2) < 3:
            continue
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False) # Welch's t-test
        
        # Calculate effect size
        effect_size = calculate_cohens_d(data1.values, data2.values)
        
        results.append({
            'feature': feature,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size_cohens_d': effect_size,
            'mean_group1': data1.mean(),
            'mean_group2': data2.mean(),
            'std_group1': data1.std(),
            'std_group2': data2.std()
        })
    
    if len(results) == 0:
        logger.warning("No valid comparisons could be performed")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Apply multiple comparison correction
    if not results_df.empty:
        from statsmodels.stats.multitest import multipletests
        reject, p_corrected, _, _ = multipletests(
            results_df['p_value'].fillna(1.0), # Fillna just in case
            alpha=config.ALPHA_LEVEL,
            method=config.MULTIPLE_COMPARISON_METHOD
        )
        results_df['p_corrected_fdr'] = p_corrected
        results_df['significant'] = reject
    
    # Sort by absolute effect size
    results_df = results_df.sort_values('effect_size_cohens_d', key=abs, ascending=False)
    
    logger.info(f"Found {(results_df['significant']).sum()} significant features")
    
    return results_df


def save_results(results: Dict, filename: str):
    """
    Save analysis results to file
    
    Parameters:
    -----------
    results : Dict
        Dictionary of results to save
    filename : str
        Output filename
    """
    filepath = config.OUTPUT_DIR / filename
    
    try:
        if filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
        elif filename.endswith('.pkl'):
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
                
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")


def create_summary_report(results: Dict, output_file: str = 'summary_report.txt'):
    """
    Create a human-readable summary report
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing analysis results
    output_file : str
        Output filename for the report
    """
    filepath = config.OUTPUT_DIR / output_file
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ADHD DIGITAL PHENOTYPE ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Data Quality Section
        if 'data_quality' in results:
            f.write("DATA QUALITY\n")
            f.write("-" * 40 + "\n")
            dq = results['data_quality']
            f.write(f"Total Subjects (after cleaning): {dq['total_subjects']}\n")
            f.write(f"\nGroup Distribution:\n")
            for group_id, count in dq['group_counts'].items():
                group_label = config.GROUP_LABELS.get(int(group_id), f"Group {group_id}")
                f.write(f"  {group_label}: {count}\n")
            f.write(f"\nFeatures with High Missing Rate: {len(dq['high_missing_features'])}\n\n")
        
        # Model Performance Section
        if 'model_results' in results and results['model_results']:
            f.write("\nMODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for comparison, metrics in results['model_results'].items():
                f.write(f"\nComparison: {comparison}\n")
                if 'logistic_regression' in metrics:
                    lr = metrics['logistic_regression']
                    f.write(f"  Logistic Regression:\n")
                    f.write(f"    Accuracy: {lr.get('accuracy', 'N/A'):.3f}\n")
                    f.write(f"    ROC-AUC: {lr.get('roc_auc', 'N/A'):.3f}\n")
                if 'random_forest' in metrics:
                    rf = metrics['random_forest']
                    f.write(f"  Random Forest:\n")
                    f.write(f"    Accuracy: {rf.get('accuracy', 'N/A'):.3f}\n")
                    f.write(f"    ROC-AUC: {rf.get('roc_auc', 'N/A'):.3f}\n")
        else:
            f.write("\nMODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write("No model results were generated.\n")
            
        # PCA Results
        if 'pca_results' in results and 'explained_variance' in results['pca_results'] and results['pca_results']['explained_variance']:
            f.write("\nPCA RESULTS\n")
            f.write("-" * 40 + "\n")
            ev = results['pca_results']['explained_variance']
            for i, var in enumerate(ev[:3], 1):
                f.write(f"  PC{i}: {var*100:.2f}% variance explained\n")
        else:
            f.write("\nPCA RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write("No PCA results were generated.\n")
            
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Report saved to: {filepath}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Summary report created: {filepath}")