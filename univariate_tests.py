"""
Univariate Statistical Tests Module
Performs feature-by-feature comparisons between groups with effect sizes and FDR correction
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict
import config
import os
import utils  # Import utils

logger = logging.getLogger(__name__)

# (FIX: Removed calculate_cohens_d, will use utils.calculate_cohens_d)

def univariate_comparison(df: pd.DataFrame,
                          features: List[str],
                          group1: int,
                          group2: int,
                          group_col: str = 'analysis_group') -> pd.DataFrame:
    """
    Perform comprehensive univariate statistical tests between two groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with residualized features
    features : List[str]
        List of features to test
    group1, group2 : int
        Group identifiers
    group_col : str
        Column containing group labels
        
    Returns:
    --------
    pd.DataFrame : Results with statistics, p-values, effect sizes
    """
    logger.info(f"Performing univariate tests: Group {group1} vs Group {group2}")
    
    group1_name = config.GROUP_LABELS[group1]
    group2_name = config.GROUP_LABELS[group2]
    
    results = []
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        # Get data for each group
        data1 = df[df[group_col] == group1][feature].dropna()
        data2 = df[df[group_col] == group2][feature].dropna()
        
        if len(data1) < 3 or len(data2) < 3:
            continue
        
        # Calculate descriptive statistics
        mean1, std1 = data1.mean(), data1.std()
        mean2, std2 = data2.mean(), data2.std()
        median1, median2 = data1.median(), data2.median()
        
        # T-test (Welch's T-test, does not assume equal variance)
        t_stat, p_ttest = stats.ttest_ind(data1, data2, equal_var=False)
        
        # Mann-Whitney U test (non-parametric alternative)
        try:
            u_stat, p_mannwhitney = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        except ValueError:
            # Can happen if data is identical
            u_stat, p_mannwhitney = np.nan, 1.0
            
        # Effect size (Cohen's d) - (FIX: Using utils function)
        cohens_d = utils.calculate_cohens_d(data1.values, data2.values)
        
        # Interpret effect size
        if pd.isna(cohens_d):
            effect_interpretation = 'nan'
        elif abs(cohens_d) < 0.2:
            effect_interpretation = 'negligible'
        elif abs(cohens_d) < 0.5:
            effect_interpretation = 'small'
        elif abs(cohens_d) < 0.8:
            effect_interpretation = 'medium'
        else:
            effect_interpretation = 'large'
        
        results.append({
            'feature': feature,
            f'{group1_name}_n': len(data1),
            f'{group1_name}_mean': mean1,
            f'{group1_name}_std': std1,
            f'{group1_name}_median': median1,
            f'{group2_name}_n': len(data2),
            f'{group2_name}_mean': mean2,
            f'{group2_name}_std': std2,
            f'{group2_name}_median': median2,
            'mean_difference': mean1 - mean2,
            't_statistic': t_stat,
            'p_ttest': p_ttest,
            'u_statistic': u_stat,
            'p_mannwhitney': p_mannwhitney,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'direction': 'higher_in_group1' if mean1 > mean2 else 'higher_in_group2'
        })
    
    if len(results) == 0:
        logger.warning("No valid comparisons performed")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # FDR correction on t-test p-values
    reject, p_corrected_ttest, _, _ = multipletests(
        results_df['p_ttest'].fillna(1.0),
        alpha=config.ALPHA_LEVEL,
        method=config.MULTIPLE_COMPARISON_METHOD
    )
    results_df['q_ttest'] = p_corrected_ttest
    results_df['significant_ttest'] = reject
    
    # FDR correction on Mann-Whitney p-values
    reject_mw, p_corrected_mw, _, _ = multipletests(
        results_df['p_mannwhitney'].fillna(1.0),
        alpha=config.ALPHA_LEVEL,
        method=config.MULTIPLE_COMPARISON_METHOD
    )
    results_df['q_mannwhitney'] = p_corrected_mw
    results_df['significant_mannwhitney'] = reject_mw
    
    # Sort by absolute effect size
    results_df = results_df.sort_values('cohens_d', key=abs, ascending=False)
    
    # Log summary
    n_sig_ttest = results_df['significant_ttest'].sum()
    n_sig_mw = results_df['significant_mannwhitney'].sum()
    
    logger.info(f"Significant features (t-test, q<{config.ALPHA_LEVEL}): {n_sig_ttest}/{len(results_df)}")
    logger.info(f"Significant features (Mann-Whitney, q<{config.ALPHA_LEVEL}): {n_sig_mw}/{len(results_df)}")
    
    # Show top effects
    logger.info("\nTop 10 features by |Cohen's d|:")
    for idx, row in results_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: d={row['cohens_d']:.3f} ({row['effect_size']}), q={row['q_ttest']:.4f}")
    
    return results_df


def run_all_univariate_tests(df: pd.DataFrame,
                             features: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Run univariate tests for all comparison pairs
    
    Parameters:
    -----------
    df : pd.DataFrame
        Residualized dataframe
    features : List[str]
        List of features
        
    Returns:
    --------
    Dict : Dictionary of comparison results
    """
    logger.info("=" * 80)
    logger.info("UNIVARIATE STATISTICAL TESTS")
    logger.info("=" * 80)
    
    all_results = {}
    
    if not features:
        logger.warning("No features provided to univariate tests. Skipping.")
        return all_results
        
    for group1_name, group2_name in config.COMPARISON_PAIRS:
        group1 = config.GROUP_DEFINITIONS[group1_name]
        group2 = config.GROUP_DEFINITIONS[group2_name]
        
        # Check if groups exist
        if group1 not in df['analysis_group'].unique():
            logger.warning(f"Group {group1_name} not found, skipping")
            continue
        if group2 not in df['analysis_group'].unique():
            logger.warning(f"Group {group2_name} not found, skipping")
            continue
        
        comparison_name = f"{config.GROUP_LABELS[group1]}_vs_{config.GROUP_LABELS[group2]}"
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Comparison: {comparison_name}")
        logger.info(f"{'=' * 60}")
        
        results_df = univariate_comparison(df, features, group1, group2)
        
        if not results_df.empty:
            all_results[comparison_name] = results_df
            
            # Save to CSV
            output_file = config.OUTPUT_DIR / f"univariate_tests_{comparison_name}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Saved results to: {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("UNIVARIATE TESTS COMPLETE")
    logger.info("=" * 80)
    
    return all_results


def create_effect_size_summary(all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create summary table of significant features across all comparisons
    
    Parameters:
    -----------
    all_results : Dict
        Dictionary of univariate test results
        
    Returns:
    --------
    pd.DataFrame : Summary table
    """
    logger.info("Creating effect size summary...")
    
    summary_rows = []
    
    for comparison, results_df in all_results.items():
        # Get significant features
        sig_features = results_df[results_df['significant_ttest']].copy()
        
        for _, row in sig_features.iterrows():
            summary_rows.append({
                'comparison': comparison,
                'feature': row['feature'],
                'cohens_d': row['cohens_d'],
                'effect_size': row['effect_size'],
                'q_value': row['q_ttest'],
                'direction': row['direction']
            })
    
    if len(summary_rows) == 0:
        logger.warning("No significant features found across any comparisons")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    output_file = config.OUTPUT_DIR / "effect_size_summary.csv"
    summary_df.to_csv(output_file, index=False)
    logger.info(f"Effect size summary saved to: {output_file}")
    
    return summary_df


if __name__ == "__main__":
    # (FIX: Removed the circular imports that caused the error)
    logger.info("This script is a module and is intended to be run by main_pipeline.py")
    pass