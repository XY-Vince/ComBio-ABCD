"""
Minimal test to isolate the exact failure point
Run this to bypass the full pipeline and test each phase independently
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phase_0_data_loading():
    """Test Phase 0: Data loading"""
    print("\n" + "="*80)
    print("TEST: PHASE 0 - DATA LOADING")
    print("="*80)
    
    try:
        from data_loader import load_and_prepare_data
        import config
        
        df, features = load_and_prepare_data()
        
        print(f"✅ Data loaded: {len(df)} subjects, {len(features)} features")
        
        # Check feature types
        print("\n🔍 Checking feature data types...")
        feature_sample = features[:5]
        for feat in feature_sample:
            if feat in df.columns:
                dtype = df[feat].dtype
                sample_val = df[feat].iloc[0]
                print(f"  {feat}: dtype={dtype}, sample={sample_val}")
        
        # Check if numeric
        numeric_count = sum([pd.api.types.is_numeric_dtype(df[f]) for f in features if f in df.columns])
        print(f"\n📊 Numeric features: {numeric_count}/{len(features)}")
        
        if numeric_count < len(features) * 0.9:
            print("❌ PROBLEM: Less than 90% of features are numeric!")
            print("   The numeric conversion fix is NOT working.")
            return False
        
        return True, df, features
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_phase_1_residualization(df, features):
    """Test Phase 1: Residualization"""
    print("\n" + "="*80)
    print("TEST: PHASE 1 - RESIDUALIZATION")
    print("="*80)
    
    try:
        from residualization import residualize_all_features
        import config
        
        df_resid, stats = residualize_all_features(
            df, features[:10],  # Test with just 10 features first
            control_group=config.GROUP_DEFINITIONS['control']
        )
        
        successful_features = [f for f in features[:10] if f in df_resid.columns]
        print(f"✅ Residualized {len(successful_features)}/10 test features")
        
        if len(successful_features) == 0:
            print("❌ PROBLEM: Zero features successfully residualized!")
            print("   Check covariates and control group data.")
            return False
        
        # Show sample statistics
        if stats:
            print("\n📊 Sample statistics:")
            for feat, stat in list(stats.items())[:3]:
                print(f"  {feat}: R²={stat['r_squared']:.4f}")
        
        return True, df_resid, successful_features
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_phase_2_modeling(df_resid, features):
    """Test Phase 2: Predictive Modeling"""
    print("\n" + "="*80)
    print("TEST: PHASE 2 - PREDICTIVE MODELING")
    print("="*80)
    
    try:
        from predictive_models import run_comparison
        import config
        
        # Test one comparison
        results = run_comparison(
            df_resid,
            features,
            group1=config.GROUP_DEFINITIONS['control'],
            group2=config.GROUP_DEFINITIONS['unmedicated_adhd'],
            tune_hyperparameters=False
        )
        
        if results:
            lr_auc = results['logistic_regression']['roc_auc']
            rf_auc = results['random_forest']['roc_auc']
            print(f"✅ Models trained successfully!")
            print(f"   LR ROC-AUC: {lr_auc:.4f}")
            print(f"   RF ROC-AUC: {rf_auc:.4f}")
            return True
        else:
            print("❌ PROBLEM: run_comparison returned empty results")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_3_pca(df_resid, features):
    """Test Phase 3: PCA"""
    print("\n" + "="*80)
    print("TEST: PHASE 3 - PCA VISUALIZATION")
    print("="*80)
    
    try:
        from visualization import run_pca_analysis
        
        results = run_pca_analysis(df_resid, features)
        
        if results and 'explained_variance' in results:
            print(f"✅ PCA completed successfully!")
            print(f"   PC1 variance: {results['explained_variance'][0]*100:.2f}%")
            return True
        else:
            print("❌ PROBLEM: PCA returned empty results")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_minimal_test():
    """Run all tests sequentially"""
    
    print("\n\n" + "="*80)
    print("MINIMAL PIPELINE TEST")
    print("Testing each phase independently to isolate failures")
    print("="*80)
    
    # Phase 0
    result = test_phase_0_data_loading()
    if not result or result[0] is False:
        print("\n❌ STOPPED: Phase 0 failed. Fix data loading first.")
        return
    
    _, df, features = result
    
    # Phase 1
    result = test_phase_1_residualization(df, features)
    if not result or result[0] is False:
        print("\n❌ STOPPED: Phase 1 failed. Fix residualization next.")
        return
    
    _, df_resid, resid_features = result
    
    # Phase 2
    result = test_phase_2_modeling(df_resid, resid_features)
    if not result:
        print("\n❌ STOPPED: Phase 2 failed. Fix modeling next.")
        return
    
    # Phase 3
    result = test_phase_3_pca(df_resid, resid_features)
    if not result:
        print("\n❌ STOPPED: Phase 3 failed. Fix PCA next.")
        return
    
    print("\n\n" + "="*80)
    print("✅ ALL PHASES PASSED!")
    print("="*80)
    print("\nYour pipeline should work now. Run:")
    print("  python main_pipeline.py")


if __name__ == "__main__":
    run_minimal_test()