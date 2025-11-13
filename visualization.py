"""
Phase 3: Visualization Module
PCA analysis and visualization of digital phenotypes
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from typing import List, Dict, Tuple
import config
import utils

logger = logging.getLogger(__name__)


def perform_pca(df: pd.DataFrame,
                features: List[str],
                n_components: int = None) -> Tuple[PCA, np.ndarray, StandardScaler]:
    """
    Perform PCA on the feature matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Residualized dataframe
    features : List[str]
        List of features to use
    n_components : int, optional
        Number of components. If None, uses config.PCA_COMPONENTS
        
    Returns:
    --------
    Tuple : PCA model, transformed data, scaler
    """
    if n_components is None:
        n_components = config.PCA_COMPONENTS
    
    logger.info(f"Performing PCA with {n_components} components...")
    
    # Prepare feature matrix
    X = df[features].copy()
    
    # Standardize features (essential for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    # Log explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    logger.info(f"PCA Results:")
    for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var), 1):
        logger.info(f"  PC{i}: {var:.4f} variance ({cum_var:.4f} cumulative)")
    
    return pca, X_pca, scaler


def plot_pca_2d(df: pd.DataFrame,
                X_pca: np.ndarray,
                pca_model: PCA,
                output_filename: str = 'pca_2d_plot'):
    """
    Create 2D PCA scatter plot colored by group
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with group labels
    X_pca : np.ndarray
        PCA-transformed data
    pca_model : PCA
        Fitted PCA model
    output_filename : str
        Output filename (without extension)
    """
    logger.info("Creating 2D PCA plot...")
    
    # Create dataframe for plotting
    df_pca = pd.DataFrame(
        X_pca[:, :2],
        columns=['PC1', 'PC2'],
        index=df.index
    )
    df_pca['group'] = df['analysis_group'].map(config.GROUP_LABELS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each group
    sns.scatterplot(
        data=df_pca,
        x='PC1',
        y='PC2',
        hue='group',
        palette=config.COLOR_PALETTE,
        s=100,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )
    
    # Labels and title
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    
    ax.set_xlabel(f'PC1 ({var1:.2f}% variance)', fontsize=14)
    ax.set_ylabel(f'PC2 ({var2:.2f}% variance)', fontsize=14)
    ax.set_title('PCA: Digital Phenotype Separation by Clinical Group', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / f"{output_filename}.{config.FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"2D PCA plot saved to {output_path}")


def plot_pca_3d(df: pd.DataFrame,
                X_pca: np.ndarray,
                pca_model: PCA,
                output_filename: str = 'pca_3d_plot'):
    """
    Create 3D PCA scatter plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with group labels
    X_pca : np.ndarray
        PCA-transformed data (must have at least 3 components)
    pca_model : PCA
        Fitted PCA model
    output_filename : str
        Output filename
    """
    if X_pca.shape[1] < 3:
        logger.warning("Need at least 3 PCA components for 3D plot")
        return
    
    logger.info("Creating 3D PCA plot...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create dataframe for plotting
    df_pca = pd.DataFrame(
        X_pca[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=df.index
    )
    df_pca['group'] = df['analysis_group'].map(config.GROUP_LABELS)
    
    # Plot each group
    groups = df_pca['group'].unique()
    colors = sns.color_palette(config.COLOR_PALETTE, n_colors=len(groups))
    
    for group, color in zip(groups, colors):
        mask = df_pca['group'] == group
        ax.scatter(
            df_pca.loc[mask, 'PC1'],
            df_pca.loc[mask, 'PC2'],
            df_pca.loc[mask, 'PC3'],
            label=group,
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5,
            c=[color]
        )
    
    # Labels
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    var3 = pca_model.explained_variance_ratio_[2] * 100
    
    ax.set_xlabel(f'PC1 ({var1:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var2:.2f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({var3:.2f}%)', fontsize=12)
    ax.set_title('3D PCA: Digital Phenotype Visualization', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    output_path = config.OUTPUT_DIR / f"{output_filename}.{config.FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"3D PCA plot saved to {output_path}")


def plot_scree(pca_model: PCA, output_filename: str = 'pca_scree_plot'):
    """
    Create scree plot showing explained variance
    
    Parameters:
    -----------
    pca_model : PCA
        Fitted PCA model with multiple components
    output_filename : str
        Output filename
    """
    logger.info("Creating scree plot...")
    
    n_components = len(pca_model.explained_variance_ratio_)
    
    explained_var = pca_model.explained_variance_ratio_[:n_components]
    cumulative_var = np.cumsum(explained_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance plot
    ax1.bar(range(1, n_components + 1), explained_var * 100, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(range(1, n_components + 1), cumulative_var * 100, 
             marker='o', linewidth=2, markersize=8, color='darkred')
    ax2.axhline(y=80, color='gray', linestyle='--', label='80% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = config.OUTPUT_DIR / f"{output_filename}.{config.FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Scree plot saved to {output_path}")


def plot_loading_heatmap(pca_model: PCA,
                         feature_names: List[str],
                         n_components: int = 5,
                         n_features: int = 30,
                         output_filename: str = 'pca_loadings'):
    """
    Create heatmap of PCA loadings for top features
    
    Parameters:
    -----------
    pca_model : PCA
        Fitted PCA model
    feature_names : List[str]
        Names of features
    n_components : int
        Number of components to show
    n_features : int
        Number of top features to show
    output_filename : str
        Output filename
    """
    logger.info("Creating PCA loading heatmap...")
    
    # Get loadings
    loadings = pca_model.components_[:n_components, :]
    
    # Find top features by absolute loading across all components
    total_loading = np.sum(np.abs(loadings), axis=0)
    top_indices = np.argsort(total_loading)[-n_features:]
    
    # Create dataframe
    loading_df = pd.DataFrame(
        loadings[:, top_indices].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=[feature_names[i] for i in top_indices]
    )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(loading_df, cmap='RdBu_r', center=0, annot=True,
                fmt='.2f', cbar_kws={'label': 'Loading'}, ax=ax, linewidth=0.5)
    ax.set_title(f'Top {n_features} Feature Loadings', fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    
    output_path = config.OUTPUT_DIR / f"{output_filename}.{config.FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loading heatmap saved to {output_path}")


def test_group_separation(df: pd.DataFrame, X_pca: np.ndarray) -> Dict:
    """
    Test statistical significance of group separation in PCA space
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with group labels
    X_pca : np.ndarray
        PCA-transformed data
        
    Returns:
    --------
    Dict : Statistical test results
    """
    logger.info("Testing statistical significance of group separation...")
    
    results = {}
    
    for pc_idx in range(min(3, X_pca.shape[1])):
        pc_name = f'PC{pc_idx + 1}'
        
        # Get data for each group
        groups_data = []
        for group_id in config.GROUP_LABELS.keys():
            mask = df['analysis_group'] == group_id
            if mask.sum() > 0:
                groups_data.append(X_pca[mask, pc_idx])
        
        # Perform ANOVA
        if len(groups_data) >= 2:
            f_stat, p_value = f_oneway(*groups_data)
            results[pc_name] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < config.ALPHA_LEVEL
            }
            
            logger.info(f"{pc_name}: F={f_stat:.4f}, p={p_value:.4f}")
    
    return results


def run_pca_analysis(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Run complete PCA analysis and visualization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Residualized dataframe
    features : List[str]
        List of features
        
    Returns:
    --------
    Dict : PCA results and statistics
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: PCA ANALYSIS - Visualizing Digital Phenotypes")
    logger.info("=" * 80)
    
    if not features:
        logger.warning("No features provided to PCA. Skipping.")
        return {}
        
    # Perform PCA with more components for scree plot
    pca_full, X_pca_full, scaler = perform_pca(
        df, features, n_components=config.PCA_N_COMPONENTS_SCREE
    )
    
    # Create visualizations
    plot_scree(pca_full, 'pca_scree_plot')
    
    # 3D plot
    plot_pca_3d(df, X_pca_full, pca_full, 'pca_3d_plot')

    # 2D plot
    plot_pca_2d(df, X_pca_full, pca_full, 'pca_2d_plot')
    
    # Loading heatmap
    plot_loading_heatmap(pca_full, features, n_components=5, n_features=30)
    
    # Test group separation
    separation_stats = test_group_separation(df, X_pca_full)
    
    logger.info("=" * 80)
    logger.info("PCA Analysis Complete")
    logger.info("=" * 80)
    
    results = {
        'pca_model': pca_full,
        'X_pca': X_pca_full,
        'scaler': scaler,
        'explained_variance': pca_full.explained_variance_ratio_,
        'separation_statistics': separation_stats
    }
    
    return results


if __name__ == "__main__":
    # (FIX: Removed the circular imports that caused the error)
    logger.info("This script is a module and is intended to be run by main_pipeline.py")
    pass