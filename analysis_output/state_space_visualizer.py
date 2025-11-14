"""
State-Space Visualization Tool

Creates publication-quality figures showing:
1. 1D projection of state space
2. 2D PCA visualization with trajectories
3. Distance distributions by group
4. SHAP summary plots
5. Consensus feature ranking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import shap
from pathlib import Path
import config

def plot_1d_state_space(positions: pd.DataFrame,
                        model_name: str = 'rf',
                        output_file: str = None):
    """
    Plot 1D projection showing where each group lands on Control→Unmedicated axis
    
    This is the KEY figure for your hypothesis
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    pos_col = f'{model_name}_relative_position'
    
    # Get groups
    control = positions[positions['analysis_group'] == 0][pos_col]
    unmedicated = positions[positions['analysis_group'] == 1][pos_col]
    stimulant = positions[positions['analysis_group'] == 2][pos_col]
    
    # Top panel: Distributions
    ax1 = axes[0]
    
    # Plot distributions
    ax1.hist(control, bins=30, alpha=0.6, color='blue', label=f'Control (n={len(control)})', density=True)
    ax1.hist(unmedicated, bins=20, alpha=0.6, color='red', label=f'Unmedicated (n={len(unmedicated)})', density=True)
    ax1.hist(stimulant, bins=20, alpha=0.6, color='purple', label=f'Stimulant (n={len(stimulant)})', density=True)
    
    # Add vertical lines for means
    ax1.axvline(control.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(unmedicated.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(stimulant.mean(), color='purple', linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Relative Position (0 = Control, 1 = Unmedicated)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title(f'State-Space Position Distribution ({model_name.upper()} Model)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # Bottom panel: Individual points (beeswarm style)
    ax2 = axes[1]
    
    groups_data = [
        (control.values, 0, 'Control', 'blue'),
        (unmedicated.values, 1, 'Unmedicated', 'red'),
        (stimulant.values, 2, 'Stimulant', 'purple')
    ]
    
    for data, y_pos, label, color in groups_data:
        # Add jitter for visibility
        y_jitter = y_pos + np.random.normal(0, 0.05, len(data))
        ax2.scatter(data, y_jitter, alpha=0.4, s=30, color=color, label=label)
        
        # Add mean marker
        ax2.scatter(data.mean(), y_pos, s=200, color=color, marker='D', 
                   edgecolors='black', linewidth=2, zorder=10)
    
    ax2.set_xlabel('Relative Position (0 = Control, 1 = Unmedicated)', fontsize=14, fontweight='bold')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Control', 'Unmedicated', 'Stimulant'], fontsize=12)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_title('Individual Subject Positions (Diamond = Group Mean)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    
    # Add annotation box
    stim_mean = stimulant.mean()
    ctrl_mean = control.mean()
    unmed_mean = unmedicated.mean()
    normalization = 100 * (1 - (stim_mean - ctrl_mean) / (unmed_mean - ctrl_mean))
    
    textstr = f'Stimulant Mean: {stim_mean:.3f}\n'
    textstr += f'Normalization: {normalization:.1f}% toward Control\n'
    textstr += f'Distance to Control: {abs(stim_mean - ctrl_mean):.3f}\n'
    textstr += f'Distance to Unmedicated: {abs(stim_mean - unmed_mean):.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    
    return fig


def plot_2d_trajectory(positions: pd.DataFrame,
                      trained_state: dict,
                      output_file: str = None):
    """
    Plot 2D PCA showing trajectory from Unmedicated → Stimulant → Control
    """
    # Perform PCA on training data
    X_train = trained_state['X_train']
    
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_train)
    
    # Transform all data
    scaler = trained_state['scaler']
    features = trained_state['features']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each group
    colors = {0: 'blue', 1: 'red', 2: 'purple'}
    labels = {0: 'Control', 1: 'Unmedicated', 2: 'Stimulant'}
    
    for group_id, color in colors.items():
        mask = positions['analysis_group'] == group_id
        group_data = positions[mask]
        
        # Get PCA coordinates for this group
        # (This requires the full dataframe, which we'd need to pass in)
        # For now, plot based on indices
        
        ax.scatter([], [], alpha=0.6, s=50, color=color, label=labels[group_id])
    
    # Compute centroids
    centroids = {}
    for group_id in [0, 1, 2]:
        mask = positions['analysis_group'] == group_id
        # Would compute PCA centroid here
        pass
    
    # Draw trajectory arrows
    # unmed_centroid → stim_centroid → control_centroid
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=14, fontweight='bold')
    ax.set_title('State-Space Trajectory: Effect of Medication', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    
    return fig


def plot_model_comparison(positions: pd.DataFrame,
                         output_file: str = None):
    """
    Compare all three models' positioning of Stimulant group
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    model_names = ['rf', 'lr', 'xgb']
    model_labels = ['Random Forest', 'Logistic Regression', 'XGBoost']
    
    for ax, model_name, model_label in zip(axes, model_names, model_labels):
        pos_col = f'{model_name}_relative_position'
        
        # Get groups
        control = positions[positions['analysis_group'] == 0][pos_col]
        unmedicated = positions[positions['analysis_group'] == 1][pos_col]
        stimulant = positions[positions['analysis_group'] == 2][pos_col]
        
        # Box plot
        data_to_plot = [control, unmedicated, stimulant]
        bp = ax.boxplot(data_to_plot, labels=['Control', 'Unmedicated', 'Stimulant'],
                       patch_artist=True)
        
        # Color boxes
        colors = ['blue', 'red', 'purple']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add individual points
        for i, (data, color) in enumerate(zip(data_to_plot, colors), 1):
            y = data
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color=color)
        
        ax.set_ylabel('Relative Position', fontsize=12, fontweight='bold')
        ax.set_title(model_label, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(-0.2, 1.2)
        
        # Add horizontal reference lines
        ax.axhline(0, color='blue', linestyle='--', alpha=0.3)
        ax.axhline(1, color='red', linestyle='--', alpha=0.3)
    
    plt.suptitle('Model Comparison: Stimulant Group Positioning', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    
    return fig


def plot_shap_summary(shap_values, X, feature_names, output_file=None):
    """
    Create SHAP summary plot showing feature importance and direction
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                     show=False, max_display=20)
    
    plt.title('Feature Importance (SHAP Values)\nRed = High Feature Value, Blue = Low Feature Value',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    
    return fig


def plot_consensus_features(feature_importance: dict,
                           top_n: int = 15,
                           output_file: str = None):
    """
    Plot consensus feature ranking across all three models
    """
    rf_imp = feature_importance['rf'].set_index('feature')['importance']
    lr_imp = feature_importance['lr'].set_index('feature')['coefficient'].abs()
    xgb_imp = feature_importance['xgb_shap'].set_index('feature')['shap_importance']
    
    # Normalize to 0-1 scale
    rf_norm = (rf_imp - rf_imp.min()) / (rf_imp.max() - rf_imp.min())
    lr_norm = (lr_imp - lr_imp.min()) / (lr_imp.max() - lr_imp.min())
    xgb_norm = (xgb_imp - xgb_imp.min()) / (xgb_imp.max() - xgb_imp.min())
    
    # Compute consensus score (average normalized importance)
    all_features = set(rf_norm.index) | set(lr_norm.index) | set(xgb_norm.index)
    
    consensus_scores = []
    for feat in all_features:
        scores = []
        if feat in rf_norm.index:
            scores.append(rf_norm[feat])
        if feat in lr_norm.index:
            scores.append(lr_norm[feat])
        if feat in xgb_norm.index:
            scores.append(xgb_norm[feat])
        
        consensus_scores.append({
            'feature': feat,
            'consensus_score': np.mean(scores),
            'n_models': len(scores)
        })
    
    consensus_df = pd.DataFrame(consensus_scores).sort_values('consensus_score', ascending=False)
    
    # Plot top N
    top_features = consensus_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_features))
    
    # Plot bars colored by number of models agreeing
    colors = ['lightcoral' if n == 1 else 'orange' if n == 2 else 'green' 
             for n in top_features['n_models']]
    
    ax.barh(y_pos, top_features['consensus_score'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Consensus Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Consensus Features Across All Models', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Important in 3 models'),
        Patch(facecolor='orange', alpha=0.7, label='Important in 2 models'),
        Patch(facecolor='lightcoral', alpha=0.7, label='Important in 1 model')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    
    return fig, consensus_df


def create_all_visualizations(results: dict, output_dir: Path):
    """
    Create all visualization figures for the state-space analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING STATE-SPACE VISUALIZATIONS")
    print("="*80)
    
    positions = results['positions']
    trained_state = results['trained_state']
    feature_interpretation = results['feature_interpretation']
    
    # Figure 1: 1D State Space (Primary result figure)
    print("\n1️⃣ Creating 1D state-space projection...")
    for model_name in ['rf', 'lr', 'xgb']:
        plot_1d_state_space(
            positions, 
            model_name=model_name,
            output_file=output_dir / f'state_space_1d_{model_name}.png'
        )
    
    # Figure 2: Model comparison
    print("\n2️⃣ Creating model comparison plot...")
    plot_model_comparison(
        positions,
        output_file=output_dir / 'model_comparison.png'
    )
    
    # Figure 3: SHAP summary
    print("\n3️⃣ Creating SHAP summary plot...")
    plot_shap_summary(
        feature_interpretation['shap_values'],
        trained_state['X_train'],
        trained_state['features'],
        output_file=output_dir / 'shap_summary.png'
    )
    
    # Figure 4: Consensus features
    print("\n4️⃣ Creating consensus features plot...")
    fig, consensus_df = plot_consensus_features(
        feature_interpretation['feature_importance'],
        top_n=20,
        output_file=output_dir / 'consensus_features.png'
    )
    
    # Save consensus DataFrame
    consensus_df.to_csv(output_dir / 'consensus_features_ranked.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS CREATED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  • state_space_1d_rf.png - Primary result (Random Forest)")
    print("  • state_space_1d_lr.png - Linear baseline")
    print("  • state_space_1d_xgb.png - Validation model")
    print("  • model_comparison.png - All three models side-by-side")
    print("  • shap_summary.png - Feature importance with directionality")
    print("  • consensus_features.png - Top features across all models")
    print("  • consensus_features_ranked.csv - Full ranking")


if __name__ == "__main__":
    print("This module should be imported and used after running state_space_analyzer.py")