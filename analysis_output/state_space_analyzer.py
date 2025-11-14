"""
State-Space Analysis Pipeline - REVISED VERSION with Train/Test Split

Key Changes:
1. Split Control+Unmedicated into 70% train / 30% test
2. Validate models on held-out test set BEFORE applying to Stimulant
3. Report test set performance to verify "ruler" works properly

Implements the 4-step framework:
1. Define Control vs Unmedicated space (train on 70%, test on 30%)
2. Create distance-based features (compute centroids)
3. Test hypothesis (position Stimulant group)
4. Interpret features (SHAP + importance)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from scipy.spatial.distance import euclidean, cdist
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import config

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: DEFINE STATE SPACE WITH TRAIN/TEST SPLIT
# ============================================================================

def train_state_space_models_with_validation(
    df: pd.DataFrame, 
    features: List[str],
    control_group: int = 0,
    unmedicated_group: int = 1,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Train three models on 70% Control+Unmedicated, validate on 30% holdout
    
    NEW: This version splits Control+Unmedicated into train/test to validate
    that the "ruler" works properly before applying to Stimulant group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset including all groups
    features : List[str]
        List of feature column names
    control_group : int
        Value for control group in analysis_group column
    unmedicated_group : int
        Value for unmedicated group in analysis_group column
    test_size : float
        Proportion to hold out for testing (default 0.3 = 30%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict : Trained models, scaler, and validation metrics
    """
    logger.info("=" * 80)
    logger.info("STEP 1: DEFINING STATE SPACE WITH TRAIN/TEST VALIDATION")
    logger.info("=" * 80)
    
    # Filter to Control + Unmedicated only
    df_train_test = df[df['analysis_group'].isin([control_group, unmedicated_group])].copy()
    
    # Create binary labels (0 = Control, 1 = Unmedicated)
    y_all = (df_train_test['analysis_group'] == unmedicated_group).astype(int)
    X_all = df_train_test[features].copy()
    
    logger.info(f"\nTotal Control + Unmedicated data:")
    logger.info(f"  Control: {(y_all == 0).sum()} subjects")
    logger.info(f"  Unmedicated ADHD: {(y_all == 1).sum()} subjects")
    logger.info(f"  Total: {len(y_all)} subjects")
    logger.info(f"  Features: {len(features)}")
    
    # ========================================================================
    # NEW: Split into train (70%) and test (30%)
    # ========================================================================
    logger.info(f"\n📊 Splitting into {int((1-test_size)*100)}% train / {int(test_size*100)}% test...")
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, 
        y_all, 
        X_all.index,  # Keep track of indices
        test_size=test_size,
        random_state=random_state,
        stratify=y_all  # Maintain class balance in both sets
    )
    
    logger.info(f"\nTraining set:")
    logger.info(f"  Control: {(y_train == 0).sum()} subjects")
    logger.info(f"  Unmedicated: {(y_train == 1).sum()} subjects")
    logger.info(f"  Total: {len(y_train)} subjects")
    
    logger.info(f"\nTest set (held-out for validation):")
    logger.info(f"  Control: {(y_test == 0).sum()} subjects")
    logger.info(f"  Unmedicated: {(y_test == 1).sum()} subjects")
    logger.info(f"  Total: {len(y_test)} subjects")
    
    # Standardize features using TRAINING set statistics only
    logger.info(f"\n🔧 Standardizing features (fit on train, transform both)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
    X_test_scaled = scaler.transform(X_test)        # Transform test using train stats
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    
    # Initialize models dictionary
    models = {}
    
    # ========================================================================
    # 1. Random Forest (Primary)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("1️⃣  RANDOM FOREST (Primary Model)")
    logger.info("="*80)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
        oob_score=True
    )
    
    logger.info("Training on 70% data...")
    rf.fit(X_train_scaled_df, y_train)
    
    # Training metrics
    train_pred = rf.predict_proba(X_train_scaled_df)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    
    # NEW: Test set validation
    logger.info("Validating on 30% held-out test set...")
    test_pred = rf.predict_proba(X_test_scaled_df)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    test_acc = accuracy_score(y_test, rf.predict(X_test_scaled_df))
    
    logger.info(f"\n  ✅ Training AUC: {train_auc:.4f}")
    logger.info(f"  ✅ Test AUC:     {test_auc:.4f}")
    logger.info(f"  ✅ Test Accuracy: {test_acc:.4f}")
    logger.info(f"  ✅ OOB Score:    {rf.oob_score_:.4f}")
    
    # Check for overfitting
    if train_auc - test_auc > 0.1:
        logger.warning(f"  ⚠️  Potential overfitting detected! Train-Test gap: {train_auc - test_auc:.3f}")
    else:
        logger.info(f"  ✅ Good generalization (Train-Test gap: {train_auc - test_auc:.3f})")
    
    models['rf'] = {
        'model': rf,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'oob_score': rf.oob_score_,
        'type': 'non-linear'
    }
    
    # ========================================================================
    # 2. Logistic Regression (Baseline)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("2️⃣  LOGISTIC REGRESSION (Baseline)")
    logger.info("="*80)
    
    lr = LogisticRegression(
        penalty='elasticnet',
        C=1.0,
        solver='saga',
        l1_ratio=0.5,
        max_iter=5000,  # Increased from 2000 to avoid convergence warnings
        random_state=random_state,
        class_weight='balanced'
    )
    
    logger.info("Training on 70% data...")
    lr.fit(X_train_scaled_df, y_train)
    
    # Training metrics
    train_pred = lr.predict_proba(X_train_scaled_df)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    
    # Test set validation
    logger.info("Validating on 30% held-out test set...")
    test_pred = lr.predict_proba(X_test_scaled_df)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    test_acc = accuracy_score(y_test, lr.predict(X_test_scaled_df))
    
    logger.info(f"\n  ✅ Training AUC: {train_auc:.4f}")
    logger.info(f"  ✅ Test AUC:     {test_auc:.4f}")
    logger.info(f"  ✅ Test Accuracy: {test_acc:.4f}")
    
    if train_auc - test_auc > 0.1:
        logger.warning(f"  ⚠️  Potential overfitting detected! Train-Test gap: {train_auc - test_auc:.3f}")
    else:
        logger.info(f"  ✅ Good generalization (Train-Test gap: {train_auc - test_auc:.3f})")
    
    models['lr'] = {
        'model': lr,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'type': 'linear',
        'coefficients': pd.DataFrame({
            'feature': features,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
    }
    
    # ========================================================================
    # 3. XGBoost (Validation)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("3️⃣  XGBOOST (Validation)")
    logger.info("="*80)
    
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, skipping...")
        models['xgb'] = None
    else:
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        logger.info("Training on 70% data...")
        xgb.fit(X_train_scaled_df, y_train)
        
        # Training metrics
        train_pred = xgb.predict_proba(X_train_scaled_df)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        # Test set validation
        logger.info("Validating on 30% held-out test set...")
        test_pred = xgb.predict_proba(X_test_scaled_df)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        test_acc = accuracy_score(y_test, xgb.predict(X_test_scaled_df))
        
        logger.info(f"\n  ✅ Training AUC: {train_auc:.4f}")
        logger.info(f"  ✅ Test AUC:     {test_auc:.4f}")
        logger.info(f"  ✅ Test Accuracy: {test_acc:.4f}")
        
        if train_auc - test_auc > 0.1:
            logger.warning(f"  ⚠️  Potential overfitting detected! Train-Test gap: {train_auc - test_auc:.3f}")
        else:
            logger.info(f"  ✅ Good generalization (Train-Test gap: {train_auc - test_auc:.3f})")
        
        models['xgb'] = {
            'model': xgb,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'type': 'non-linear'
        }
    
    # ========================================================================
    # Model Comparison
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON (Test Set Performance)")
    logger.info("="*80)
    logger.info(f"Random Forest:        AUC={models['rf']['test_auc']:.4f}, Acc={models['rf']['test_accuracy']:.4f}")
    logger.info(f"Logistic Regression:  AUC={models['lr']['test_auc']:.4f}, Acc={models['lr']['test_accuracy']:.4f}")
    if XGBOOST_AVAILABLE and models['xgb'] is not None:
        logger.info(f"XGBoost:              AUC={models['xgb']['test_auc']:.4f}, Acc={models['xgb']['test_accuracy']:.4f}")
    
    # Decision: Use best model for downstream analysis
    best_model_name = max(models.keys(), 
                          key=lambda k: models[k]['test_auc'] if models[k] is not None else 0)
    logger.info(f"\n🏆 Best model: {best_model_name.upper()} (Test AUC: {models[best_model_name]['test_auc']:.4f})")
    
    # ========================================================================
    # Return all information
    # ========================================================================
    return {
        'models': models,
        'scaler': scaler,
        'features': features,
        'X_train': X_train_scaled_df,
        'y_train': y_train,
        'X_test': X_test_scaled_df,
        'y_test': y_test,
        'train_indices': idx_train,
        'test_indices': idx_test,
        'best_model': best_model_name
    }


# ============================================================================
# STEP 2: COMPUTE STATE-SPACE DISTANCES (UNCHANGED FROM ORIGINAL)
# ============================================================================

def compute_state_space_positions(trained_state: Dict,
                                  df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Compute position of ALL subjects in the state space defined by Step 1
    
    Key: Use the FROZEN models from Step 1. Do not retrain.
    
    Returns:
        DataFrame with distances for each subject
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: COMPUTING STATE-SPACE POSITIONS")
    logger.info("=" * 80)
    
    models = trained_state['models']
    scaler = trained_state['scaler']
    features = trained_state['features']
    
    # Prepare all data (including Stimulant group)
    X_all = df_all[features].copy()
    X_all_scaled = scaler.transform(X_all)  # Use training scaler
    X_all_scaled_df = pd.DataFrame(X_all_scaled, columns=features, index=X_all.index)
    
    logger.info(f"Computing positions for {len(df_all)} subjects")
    
    results = pd.DataFrame(index=df_all.index)
    results['analysis_group'] = df_all['analysis_group']
    
    # For each model, compute probabilities and relative positions
    for model_name, model_info in models.items():
        if model_info is None:
            continue
            
        model = model_info['model']
        
        logger.info(f"\n{model_name.upper()} predictions...")
        
        # Get prediction probabilities
        probs = model.predict_proba(X_all_scaled_df)
        prob_control = probs[:, 0]
        prob_unmed = probs[:, 1]
        
        # Compute distances to group centroids
        # Training set centroids
        train_control_mean = prob_control[results['analysis_group'] == 0].mean()
        train_unmed_mean = prob_unmed[results['analysis_group'] == 1].mean()
        
        # Distance for each subject
        dist_to_control = np.abs(prob_control - train_control_mean)
        dist_to_unmed = np.abs(prob_unmed - train_unmed_mean)
        
        # Relative position (0 = Control-like, 1 = Unmedicated-like)
        relative_position = dist_to_control / (dist_to_control + dist_to_unmed + 1e-10)
        
        # Store results
        results[f'{model_name}_prob_unmed'] = prob_unmed
        results[f'{model_name}_prob_control'] = prob_control
        results[f'{model_name}_dist_to_control'] = dist_to_control
        results[f'{model_name}_dist_to_unmed'] = dist_to_unmed
        results[f'{model_name}_relative_position'] = relative_position
        
        logger.info(f"  ✅ Computed positions for all subjects")
    
    return results


# ============================================================================
# STEP 3: ANALYZE MEDICATION EFFECT (UNCHANGED FROM ORIGINAL)
# ============================================================================

def analyze_medication_effect(positions: pd.DataFrame,
                              stimulant_group: int = 2) -> Dict:
    """
    Analyze where Stimulant group falls in the state space
    
    Tests hypothesis: Stimulant group should be intermediate between
    Control and Unmedicated groups
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: HYPOTHESIS TESTING - MEDICATION EFFECT")
    logger.info("=" * 80)
    
    results = {}
    
    # For each model
    model_names = [col.replace('_relative_position', '') 
                   for col in positions.columns 
                   if col.endswith('_relative_position')]
    
    for model_name in model_names:
        logger.info(f"\n{'='*80}")
        logger.info(f"{model_name.upper()} Analysis")
        logger.info(f"{'='*80}")
        
        pos_col = f'{model_name}_relative_position'
        
        # Extract groups
        control = positions[positions['analysis_group'] == 0]
        unmedicated = positions[positions['analysis_group'] == 1]
        stimulant = positions[positions['analysis_group'] == stimulant_group]
        
        pos_control = control[pos_col].values
        pos_unmed = unmedicated[pos_col].values
        pos_stim = stimulant[pos_col].values
        
        # Compute means
        mean_control = pos_control.mean()
        mean_unmed = pos_unmed.mean()
        mean_stim = pos_stim.mean()
        
        logger.info(f"\nMean relative positions (0=Control, 1=Unmedicated):")
        logger.info(f"  Control:     {mean_control:.3f}")
        logger.info(f"  Unmedicated: {mean_unmed:.3f}")
        logger.info(f"  Stimulant:   {mean_stim:.3f}")
        
        # Test H1: Stimulant is intermediate
        if mean_control < mean_stim < mean_unmed or mean_unmed < mean_stim < mean_control:
            logger.info(f"\n✅ H1 SUPPORTED: Stimulant group is INTERMEDIATE")
            
            # How much normalized?
            normalization_pct = 100 * (1 - abs(mean_stim - mean_control) / abs(mean_unmed - mean_control))
            logger.info(f"  Normalization: {normalization_pct:.1f}% toward Control")
        else:
            logger.info(f"\n❌ H1 NOT SUPPORTED: Stimulant group is NOT intermediate")
            normalization_pct = 0
        
        # Statistical tests
        u_stat_ctrl, p_ctrl = mannwhitneyu(pos_stim, pos_control, alternative='two-sided')
        u_stat_unmed, p_unmed = mannwhitneyu(pos_stim, pos_unmed, alternative='two-sided')
        
        dist_to_control = np.abs(pos_stim - mean_control).mean()
        dist_to_unmed = np.abs(pos_stim - mean_unmed).mean()
        
        logger.info(f"\nStatistical Tests:")
        logger.info(f"  Stimulant vs Control:     p = {p_ctrl:.4f}")
        logger.info(f"  Stimulant vs Unmedicated: p = {p_unmed:.4f}")
        logger.info(f"\nDistance Metrics:")
        logger.info(f"  Mean distance to Control:     {dist_to_control:.3f}")
        logger.info(f"  Mean distance to Unmedicated: {dist_to_unmed:.3f}")
        
        if dist_to_control < dist_to_unmed:
            logger.info(f"  → Stimulant group is CLOSER to Control")
        else:
            logger.info(f"  → Stimulant group is CLOSER to Unmedicated")
        
        results[model_name] = {
            'mean_control': mean_control,
            'mean_unmedicated': mean_unmed,
            'mean_stimulant': mean_stim,
            'normalization_pct': normalization_pct,
            'p_stim_vs_control': p_ctrl,
            'p_stim_vs_unmedicated': p_unmed,
            'dist_to_control': dist_to_control,
            'dist_to_unmed': dist_to_unmed,
            'closer_to': 'control' if dist_to_control < dist_to_unmed else 'unmedicated'
        }
    
    return results


# ============================================================================
# STEP 4: INTERPRET FEATURES (UNCHANGED FROM ORIGINAL - but skip if no SHAP)
# ============================================================================

def interpret_state_space_features(trained_state: Dict,
                                   df_all: pd.DataFrame,
                                   top_n: int = 20) -> Dict:
    """
    Identify which features define the state space
    
    Uses three methods:
    1. RF feature importance
    2. XGBoost SHAP values (if available)
    3. LR coefficients
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: FEATURE INTERPRETATION")
    logger.info("=" * 80)
    
    models = trained_state['models']
    features = trained_state['features']
    X_train = trained_state['X_train']
    
    feature_importance = {}
    
    # 1. Random Forest
    logger.info("\n1️⃣  Random Forest Feature Importance...")
    rf_model = models['rf']['model']
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance['rf'] = rf_importance
    
    logger.info(f"\nTop 10 RF Features:")
    for idx, row in rf_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    # 2. Logistic Regression
    logger.info(f"\n2️⃣  Logistic Regression Coefficients...")
    lr_coef = models['lr']['coefficients']
    feature_importance['lr'] = lr_coef
    
    logger.info(f"\nTop 10 LR Features (by |coefficient|):")
    for idx, row in lr_coef.head(10).iterrows():
        direction = "→ Unmedicated" if row['coefficient'] > 0 else "→ Control"
        logger.info(f"  {row['feature']:<40} {row['coefficient']:>7.4f} {direction}")
    
    # 3. XGBoost SHAP (if available)
    if SHAP_AVAILABLE and models.get('xgb') is not None:
        logger.info(f"\n3️⃣  XGBoost SHAP Values...")
        xgb_model = models['xgb']['model']
        
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_train)
            
            shap_importance = pd.DataFrame({
                'feature': features,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            feature_importance['xgb_shap'] = shap_importance
            
            logger.info(f"\nTop 10 XGB SHAP Features:")
            for idx, row in shap_importance.head(10).iterrows():
                logger.info(f"  {row['feature']:<40} {row['shap_importance']:.4f}")
        except Exception as e:
            logger.warning(f"  ⚠️  SHAP computation failed: {str(e)}")
            shap_values = None
            explainer = None
    else:
        logger.info(f"\n3️⃣  XGBoost SHAP Values... SKIPPED (SHAP not available)")
        shap_values = None
        explainer = None
    
    # Find consensus features
    logger.info("\n" + "="*80)
    logger.info(f"CONSENSUS FEATURES (Top {top_n} in at least 2 models)")
    logger.info("="*80)
    
    rf_top = set(rf_importance.head(top_n)['feature'])
    lr_top = set(lr_coef.head(top_n)['feature'])
    
    if 'xgb_shap' in feature_importance:
        xgb_top = set(feature_importance['xgb_shap'].head(top_n)['feature'])
        consensus = [feat for feat in features 
                    if sum([feat in rf_top, feat in lr_top, feat in xgb_top]) >= 2]
    else:
        consensus = [feat for feat in features 
                    if sum([feat in rf_top, feat in lr_top]) >= 2]
    
    logger.info(f"\n{len(consensus)} features appear in top {top_n} of 2+ models:")
    for feat in consensus[:15]:
        logger.info(f"  • {feat}")
    
    return {
        'feature_importance': feature_importance,
        'consensus_features': consensus,
        'explainer': explainer,
        'shap_values': shap_values
    }


# ============================================================================
# MAIN PIPELINE (UPDATED TO USE NEW TRAINING FUNCTION)
# ============================================================================

def run_state_space_analysis(df: pd.DataFrame,
                             features: List[str],
                             output_dir: Path = None,
                             test_size: float = 0.3) -> Dict:
    """
    Run complete state-space analysis pipeline with train/test validation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with all groups
    features : List[str]
        List of feature column names
    output_dir : Path
        Directory to save outputs
    test_size : float
        Proportion for test set (default 0.3 = 30%)
    
    Returns:
    --------
    Dict : All results including validation metrics
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / 'state_space_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("STATE-SPACE ANALYSIS PIPELINE (WITH VALIDATION)")
    logger.info("="*80)
    
    # Step 1: Train models on 70%, validate on 30%
    trained_state = train_state_space_models_with_validation(
        df, features, test_size=test_size
    )
    
    # Step 2: Compute positions for all subjects (including Stimulant)
    positions = compute_state_space_positions(trained_state, df)
    
    # Save positions
    positions.to_csv(output_dir / 'state_space_positions.csv')
    logger.info(f"\n✅ Saved positions to: {output_dir / 'state_space_positions.csv'}")
    
    # Step 3: Analyze medication effect on Stimulant group
    medication_effect = analyze_medication_effect(positions)
    
    # Step 4: Interpret features
    feature_interpretation = interpret_state_space_features(trained_state, df)
    
    # Save consensus features
    consensus_df = pd.DataFrame({'feature': feature_interpretation['consensus_features']})
    consensus_df.to_csv(output_dir / 'consensus_features.csv', index=False)
    
    # Create validation summary
    validation_summary = pd.DataFrame([
        {
            'model': name,
            'train_auc': info['train_auc'] if info is not None else np.nan,
            'test_auc': info['test_auc'] if info is not None else np.nan,
            'test_accuracy': info['test_accuracy'] if info is not None else np.nan,
            'overfitting_gap': info['train_auc'] - info['test_auc'] if info is not None else np.nan
        }
        for name, info in trained_state['models'].items()
        if info is not None
    ])
    validation_summary.to_csv(output_dir / 'validation_metrics.csv', index=False)
    logger.info(f"✅ Saved validation metrics to: {output_dir / 'validation_metrics.csv'}")
    
    return {
        'trained_state': trained_state,
        'positions': positions,
        'medication_effect': medication_effect,
        'feature_interpretation': feature_interpretation,
        'validation_summary': validation_summary
    }


if __name__ == "__main__":
    logger.info("This module should be imported and used by main_pipeline.py")