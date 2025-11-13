"""
Deep Diagnostic: Inspect exactly what's happening in residualize_feature()

This will show us:
1. What dtype y_controls actually has when it reaches OLS
2. What dtype X_controls has
3. Where exactly the conversion is failing
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_residualization_file():
    """Check what the residualize_feature function actually looks like"""
    
    file_path = Path('residualization.py')
    
    if not file_path.exists():
        print("❌ residualization.py not found!")
        return
    
    print("="*80)
    print("INSPECTING residualize_feature() FUNCTION")
    print("="*80)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the function
    in_function = False
    function_lines = []
    line_numbers = []
    
    for i, line in enumerate(lines, 1):
        if 'def residualize_feature(' in line:
            in_function = True
        
        if in_function:
            function_lines.append(line)
            line_numbers.append(i)
            
            # Stop at next function definition or end of significant indentation
            if i > line_numbers[0] and line.startswith('def ') and 'residualize_feature' not in line:
                break
    
    if not function_lines:
        print("❌ Could not find residualize_feature() function!")
        return
    
    print(f"\n📍 Function found at lines {line_numbers[0]}-{line_numbers[-1]}")
    print("\n🔍 Checking for fixes:")
    print("-"*80)
    
    function_text = ''.join(function_lines)
    
    # Check each fix
    fixes_status = {
        'Fix 1: y_controls numeric conversion': 'pd.to_numeric(y_controls' in function_text,
        'Fix 2: X_controls numeric conversion': 'for col in X_controls.columns:' in function_text and 'X_controls[col] = pd.to_numeric' in function_text,
        'Fix 3: X_all numeric conversion': 'for col in X_all.columns:' in function_text and 'X_all[col] = pd.to_numeric' in function_text,
        'Fix 4: feature_values conversion': 'feature_values = pd.to_numeric(df_all[feature_name]' in function_text,
    }
    
    all_applied = True
    for fix_name, is_applied in fixes_status.items():
        status = "✅" if is_applied else "❌"
        print(f"{status} {fix_name}")
        if not is_applied:
            all_applied = False
    
    if not all_applied:
        print("\n⚠️  NOT ALL FIXES WERE APPLIED!")
        print("\nLet me show you the relevant sections:")
        print("="*80)
        
        # Show the sections that need fixing
        for i, (line_num, line) in enumerate(zip(line_numbers, function_lines)):
            # Show context around key lines
            if 'y_controls = df_controls[feature_name].dropna()' in line:
                print(f"\n📍 Line {line_num} (y_controls extraction):")
                for j in range(max(0, i-1), min(len(function_lines), i+3)):
                    print(f"  {line_numbers[j]:4d}: {function_lines[j]}", end='')
            
            elif 'X_controls = df_controls.loc[y_controls.index, covariate_cols].dropna()' in line:
                print(f"\n📍 Line {line_num} (X_controls extraction):")
                for j in range(max(0, i-1), min(len(function_lines), i+3)):
                    print(f"  {line_numbers[j]:4d}: {function_lines[j]}", end='')
            
            elif 'X_all = df_all[covariate_cols].copy()' in line:
                print(f"\n📍 Line {line_num} (X_all extraction):")
                for j in range(max(0, i-1), min(len(function_lines), i+3)):
                    print(f"  {line_numbers[j]:4d}: {function_lines[j]}", end='')
            
            elif 'residuals = ' in line and 'df_all[feature_name]' in line:
                print(f"\n📍 Line {line_num} (residuals calculation):")
                for j in range(max(0, i-1), min(len(function_lines), i+3)):
                    print(f"  {line_numbers[j]:4d}: {function_lines[j]}", end='')
    
    return all_applied, function_text


def test_actual_residualization():
    """Test residualization with detailed dtype logging"""
    
    print("\n\n" + "="*80)
    print("TESTING ACTUAL RESIDUALIZATION WITH DEBUG INFO")
    print("="*80)
    
    try:
        from data_loader import load_and_prepare_data
        import config
        
        # Load data
        print("\n1️⃣ Loading data...")
        df, features = load_and_prepare_data()
        print(f"✅ Loaded {len(features)} features")
        
        # Test residualization on ONE feature with detailed logging
        from residualization import residualize_all_features
        from data_loader import prepare_covariates
        
        print("\n2️⃣ Preparing covariates...")
        df_prepared = prepare_covariates(df)
        
        # Force numeric on test feature
        test_feature = features[0]
        print(f"\n3️⃣ Testing with feature: {test_feature}")
        
        df_prepared[test_feature] = pd.to_numeric(df_prepared[test_feature], errors='coerce')
        df_prepared[test_feature] = df_prepared[test_feature].astype('float64')
        
        print(f"   df_prepared[{test_feature}].dtype = {df_prepared[test_feature].dtype}")
        
        # Get controls
        control_group = config.GROUP_DEFINITIONS['control']
        df_controls = df_prepared[df_prepared['analysis_group'] == control_group].copy()
        
        print(f"\n4️⃣ After extracting controls:")
        print(f"   df_controls[{test_feature}].dtype = {df_controls[test_feature].dtype}")
        
        # Try extracting y_controls
        y_controls = df_controls[test_feature].dropna()
        print(f"\n5️⃣ After extracting y_controls:")
        print(f"   y_controls.dtype = {y_controls.dtype}")
        print(f"   type(y_controls) = {type(y_controls)}")
        print(f"   Sample values: {y_controls.head(3).tolist()}")
        
        # Try forcing numeric
        print(f"\n6️⃣ Forcing to numeric:")
        y_controls_numeric = pd.to_numeric(y_controls, errors='coerce')
        print(f"   After pd.to_numeric: dtype = {y_controls_numeric.dtype}")
        
        # Try creating numpy array
        print(f"\n7️⃣ Converting to numpy:")
        try:
            y_array = np.asarray(y_controls_numeric)
            print(f"   ✅ Success! dtype = {y_array.dtype}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Now check covariates
        print(f"\n8️⃣ Checking covariates:")
        from residualization import get_covariate_columns
        covariate_cols = get_covariate_columns(df_prepared)
        print(f"   Covariates: {covariate_cols}")
        
        X_controls = df_controls.loc[y_controls.index, covariate_cols].dropna()
        for col in covariate_cols:
            print(f"   {col}: dtype = {X_controls[col].dtype}")
        
        print("\n9️⃣ THE PROBLEM:")
        print("   If any of the above dtypes show 'object', that's where the issue is.")
        print("   The fixes need to be applied AFTER each extraction.")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    all_fixes_applied, _ = inspect_residualization_file()
    
    if not all_fixes_applied:
        print("\n\n💡 RECOMMENDATION:")
        print("="*80)
        print("The automatic patcher missed some fixes.")
        print("You need to manually edit residualization.py")
        print("\nI'll create a fully corrected version for you to copy-paste.")
    
    # Run the actual test
    test_actual_residualization()