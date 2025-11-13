"""
Phase 2: Predictive Modeling Module
Implements Logistic Regression and Random Forest for group classification
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
# (FIX: Removed unused imports: SelectKBest, f_classif)
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, accuracy_score, precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import config
import utils

logger = logging.getLogger(__name__)


def prepare_model_data(df: pd.DataFrame,
                      features: List[str],
                      group1: int,
                      group2: int) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare data for binary classification between two groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Residualized dataframe
    features : List[str]
        List of feature columns
    group1, group2 : int
        Group identifiers to compare
        
    Returns:
    --------
    Tuple : X (features), y (labels), feature names
    """
    logger.info(f"Preparing data for comparison: Group {group1} vs Group {group2}")
    
    # Filter to only include the two groups of interest
    df_subset = df[df['analysis_group'].isin([group1, group2])].copy()
    
    # Convert to binary labels (0 and 1)
    df_subset['binary_label'] = (df_subset['analysis_group'] == group2).astype(int)
    
    # Select features
    available_features = [f for f in features if f in df_subset.columns]
    X = df_subset[available_features].copy()
    y = df_subset['binary_label']
    
    # Handle empty X (if no features were passed)
    if X.empty:
        logger.error("No features provided to prepare_model_data.")
        return X, y, []
        
    # Remove features with too many missing values or zero variance
    selector = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    
    X_filtered = pd.DataFrame(
        selector.fit_transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index
    )
    
    final_features = list(X_filtered.columns)
    
    logger.info(f"Data prepared: {len(X_filtered)} samples, {len(final_features)} features")
    logger.info(f"  Group {group1}: {(y == 0).sum()} samples")
    logger.info(f"  Group {group2}: {(y == 1).sum()} samples")
    
    return X_filtered, y, final_features


def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              feature_names: List[str],
                              tune_hyperparameters: bool = False) -> Dict:
    """
    Train and evaluate Logistic Regression model
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test labels
    feature_names : List[str]
        Names of features
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    Dict : Model results and metrics
    """
    logger.info("Training Logistic Regression model...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        lr = LogisticRegression(solver='saga', max_iter=2000, random_state=config.RANDOM_STATE, class_weight='balanced', penalty='elasticnet')
        grid_search = GridSearchCV(
            lr, config.LOGISTIC_GRID,
            cv=config.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
    else:
        model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
        model.fit(X_train_scaled, y_train)
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    # Feature importance (coefficients)
    coefficients = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Logistic Regression Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    results = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'coefficients': coefficients,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return results


def train_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       feature_names: List[str],
                       tune_hyperparameters: bool = False) -> Dict:
    """
    Train and evaluate Random Forest model
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test labels
    feature_names : List[str]
        Names of features
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    Dict : Model results and metrics
    """
    logger.info("Training Random Forest model...")
    
    # Initialize model
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning (this may take a while)...")
        rf = RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced')
        grid_search = GridSearchCV(
            rf, config.RF_GRID,
            cv=config.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
    else:
        model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        model.fit(X_train, y_train)
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Random Forest Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    results = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'feature_importances': importances,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return results


def run_comparison(df: pd.DataFrame,
                  features: List[str],
                  group1: int,
                  group2: int,
                  tune_hyperparameters: bool = False) -> Dict:
    """
    Run complete comparison between two groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Residualized dataframe
    features : List[str]
        List of features
    group1, group2 : int
        Groups to compare
    tune_hyperparameters : bool
        Whether to tune hyperparameters
        
    Returns:
    --------
    Dict : Complete results for both models
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 2: PREDICTIVE MODELING")
    logger.info(f"Comparing: {config.GROUP_LABELS[group1]} vs {config.GROUP_LABELS[group2]}")
    logger.info("=" * 80)
    
    # Prepare data
    X, y, feature_names = prepare_model_data(df, features, group1, group2)
    
    if len(feature_names) == 0:
        logger.error("No features available for modeling. Skipping comparison.")
        return {}

    # Check class balance
    class_counts = y.value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    if class_counts.min() < config.MIN_SUBJECTS_PER_GROUP:
        logger.warning(f"Small sample size in one group ({class_counts.min()}). Results may be unreliable.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train models
    lr_results = train_logistic_regression(
        X_train, y_train, X_test, y_test,
        feature_names, tune_hyperparameters
    )
    
    rf_results = train_random_forest(
        X_train, y_train, X_test, y_test,
        feature_names, tune_hyperparameters
    )
    
    # Compare models
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<20} {'Logistic Reg':<15} {'Random Forest':<15}")
    logger.info("-" * 80)
    logger.info(f"{'Accuracy':<20} {lr_results['accuracy']:<15.4f} {rf_results['accuracy']:<15.4f}")
    logger.info(f"{'ROC-AUC':<20} {lr_results['roc_auc']:<15.4f} {rf_results['roc_auc']:<15.4f}")
    logger.info(f"{'Precision':<20} {lr_results['precision']:<15.4f} {rf_results['precision']:<15.4f}")
    logger.info(f"{'Recall':<20} {lr_results['recall']:<15.4f} {rf_results['recall']:<15.4f}")
    logger.info(f"{'F1-Score':<20} {lr_results['f1_score']:<15.4f} {rf_results['f1_score']:<15.4f}")
    logger.info("=" * 80)
    
    results = {
        'comparison': f"{config.GROUP_LABELS[group1]}_vs_{config.GROUP_LABELS[group2]}",
        'group1': group1,
        'group2': group2,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'logistic_regression': lr_results,
        'random_forest': rf_results,
        'feature_names': feature_names
    }
    
    return results


def plot_model_results(results: Dict, output_prefix: str):
    """
    Create visualization plots for model results
    
    Parameters:
    -----------
    results : Dict
        Model results from run_comparison
    output_prefix : str
        Prefix for output filenames
    """
    logger.info("Creating visualization plots...")
    
    comparison_name = results['comparison']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ROC Curves
    ax1 = plt.subplot(2, 4, 1)
    
    # Logistic Regression ROC
    lr_results = results['logistic_regression']
    fpr_lr, tpr_lr, _ = roc_curve(lr_results['y_test'], lr_results['y_pred_proba'])
    ax1.plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC={lr_results['roc_auc']:.3f})", linewidth=2)
    
    # Random Forest ROC
    rf_results = results['random_forest']
    fpr_rf, tpr_rf, _ = roc_curve(rf_results['y_test'], rf_results['y_pred_proba'])
    ax1.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_results['roc_auc']:.3f})", linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Chance', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # 2. Confusion Matrix - Logistic Regression
    ax2 = plt.subplot(2, 4, 2)
    sns.heatmap(lr_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                cbar=False, ax=ax2)
    ax2.set_title('Confusion Matrix\nLogistic Regression', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. Confusion Matrix - Random Forest
    ax3 = plt.subplot(2, 4, 3)
    sns.heatmap(rf_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                cbar=False, ax=ax3)
    ax3.set_title('Confusion Matrix\nRandom Forest', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. Feature Importance - Top 20 (Random Forest)
    ax4 = plt.subplot(2, 4, 4)
    top_features_rf = rf_results['feature_importances'].head(20)
    ax4.barh(range(len(top_features_rf)), top_features_rf['importance'])
    ax4.set_yticks(range(len(top_features_rf)))
    ax4.set_yticklabels(top_features_rf['feature'], fontsize=8)
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title('Top 20 Features\nRandom Forest', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Feature Coefficients - Top 20 (Logistic Regression)
    ax5 = plt.subplot(2, 4, 5)
    top_pos_coef = lr_results['coefficients'].nlargest(20, 'coefficient')
    top_neg_coef = lr_results['coefficients'].nsmallest(20, 'coefficient')
    top_coef = pd.concat([top_pos_coef, top_neg_coef]).sort_values('coefficient', ascending=False)
    
    colors = ['green' if x > 0 else 'red' for x in top_coef['coefficient']]
    ax5.barh(range(len(top_coef)), top_coef['coefficient'], color=colors)
    ax5.set_yticks(range(len(top_coef)))
    ax5.set_yticklabels(top_coef['feature'], fontsize=8)
    ax5.set_xlabel('Coefficient', fontsize=12)
    ax5.set_title('Top 20 Pos/Neg Coefficients\nLogistic Regression', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Cross-Validation Scores
    ax6 = plt.subplot(2, 4, 6)
    cv_data = pd.DataFrame({
        'Logistic Regression': lr_results['cv_scores'],
        'Random Forest': rf_results['cv_scores']
    })
    sns.boxplot(data=cv_data, ax=ax6, palette=config.COLOR_PALETTE)
    ax6.set_ylabel('ROC-AUC Score', fontsize=12)
    ax6.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Model Performance Comparison
    ax7 = plt.subplot(2, 4, 7)
    metrics = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    lr_scores = [lr_results['accuracy'], lr_results['roc_auc'], lr_results['precision'],
                 lr_results['recall'], lr_results['f1_score']]
    rf_scores = [rf_results['accuracy'], rf_results['roc_auc'], rf_results['precision'],
                 rf_results['recall'], rf_results['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax7.bar(x - width/2, lr_scores, width, label='Logistic Reg', alpha=0.8)
    ax7.bar(x + width/2, rf_scores, width, label='Random Forest', alpha=0.8)
    ax7.set_ylabel('Score', fontsize=12)
    ax7.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics, rotation=45, ha='right')
    ax7.legend()
    ax7.set_ylim([0, 1])
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Predicted Probability Distribution
    ax8 = plt.subplot(2, 4, 8)
    sns.histplot(lr_results['y_pred_proba'][lr_results['y_test'] == 0], 
                 bins=20, alpha=0.5, label='Class 0 (G1)', color='blue', ax=ax8)
    sns.histplot(lr_results['y_pred_proba'][lr_results['y_test'] == 1],
                 bins=20, alpha=0.5, label='Class 1 (G2)', color='red', ax=ax8)
    ax8.set_xlabel('Predicted Probability (Class 1)', fontsize=12)
    ax8.set_ylabel('Count', fontsize=12)
    ax8.set_title('Predicted Probability Distribution\nLogistic Regression', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    plt.suptitle(f'Model Results: {comparison_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = config.OUTPUT_DIR / f"{output_prefix}_{comparison_name}_results.{config.FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_file}")


if __name__ == "__main__":
    # (FIX: Removed the circular imports that caused the error)
    logger.info("This script is a module and is intended to be run by main_pipeline.py")
    pass