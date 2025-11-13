"""
Diagnostic Script to Show Exactly What's Happening with Data Loading
Run this to see sample sizes at each step
"""

import pandas as pd
import config

def diagnose_data_loading():
    """
    Show detailed breakdown of data loading process
    """
    
    print("=" * 80)
    print("FITBIT DATA LOADING DIAGNOSTICS")
    print("=" * 80)
    
    # Step 1: Load raw Excel file
    print(f"\n📂 Loading file: {config.FITBIT_FILE}")
    excel_file = pd.ExcelFile(config.FITBIT_FILE)
    sheet_names = excel_file.sheet_names
    
    print(f"\n✓ Found {len(sheet_names)} sheets:")
    for sheet in sheet_names:
        print(f"   - {sheet}")
    
    print("\n" + "=" * 80)
    print("STEP 1: RAW DATA FROM EACH SHEET")
    print("=" * 80)
    
    all_dfs = []
    total_raw = 0
    
    for sheet_name in sheet_names:
        df_sheet = pd.read_excel(config.FITBIT_FILE, sheet_name=sheet_name)
        
        # Determine group assignment
        if 'control' in sheet_name.lower() or 'nc' in sheet_name.lower():
            group_name = "NC Controls (Group 0)"
            group_id = 0
        elif 'unmedicated' in sheet_name.lower():
            group_name = "ADHD Unmedicated (Group 1)"
            group_id = 1
        elif 'stimulant' in sheet_name.lower():
            group_name = "ADHD Stimulants (Group 2)"
            group_id = 2
        else:
            group_name = "UNKNOWN - Defaulting to Controls"
            group_id = 0
        
        df_sheet['analysis_group'] = group_id
        df_sheet['source_sheet'] = sheet_name
        
        print(f"\n📊 Sheet: '{sheet_name}'")
        print(f"   Assigned to: {group_name}")
        print(f"   Raw rows: {len(df_sheet)}")
        print(f"   Columns: {len(df_sheet.columns)}")
        
        # Check for subjectkey
        if 'subjectkey' in df_sheet.columns:
            unique_subjects = df_sheet['subjectkey'].nunique()
            print(f"   ✓ Has 'subjectkey' column")
            print(f"   Unique subjects: {unique_subjects}")
        else:
            print(f"   ✗ Missing 'subjectkey' column!")
        
        all_dfs.append(df_sheet)
        total_raw += len(df_sheet)
    
    # Step 2: Combined data
    print("\n" + "=" * 80)
    print("STEP 2: COMBINED DATA (All sheets merged)")
    print("=" * 80)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows after combining: {len(df_combined)}")
    print(f"Expected total: {total_raw}")
    
    group_counts = df_combined['analysis_group'].value_counts().sort_index()
    print("\nGroup distribution:")
    for group_id, count in group_counts.items():
        group_label = config.GROUP_LABELS.get(group_id, f"Group {group_id}")
        print(f"   {group_label}: {count} subjects")
    
    # Step 3: Check for covariates
    print("\n" + "=" * 80)
    print("STEP 3: COVARIATE AVAILABILITY")
    print("=" * 80)
    
    required_covariates = config.COVARIATES
    print(f"\nRequired covariates: {required_covariates}")
    
    for cov in required_covariates:
        if cov in df_combined.columns:
            missing_count = df_combined[cov].isna().sum()
            missing_pct = (missing_count / len(df_combined)) * 100
            print(f"   ✓ {cov}: {len(df_combined) - missing_count} complete ({missing_pct:.1f}% missing)")
        else:
            print(f"   ✗ {cov}: NOT FOUND in data")
    
    # Step 4: Check feature availability
    print("\n" + "=" * 80)
    print("STEP 4: FITBIT FEATURES")
    print("=" * 80)
    
    available_features = [f for f in config.ALL_FITBIT_FEATURES if f in df_combined.columns]
    missing_features = [f for f in config.ALL_FITBIT_FEATURES if f not in df_combined.columns]
    
    print(f"\nConfigured features: {len(config.ALL_FITBIT_FEATURES)}")
    print(f"Available in data: {len(available_features)}")
    print(f"Missing from data: {len(missing_features)}")
    
    if missing_features:
        print(f"\nFirst 10 missing features:")
        for feat in missing_features[:10]:
            print(f"   - {feat}")
    
    # Step 5: Simulate missing value handling
    print("\n" + "=" * 80)
    print("STEP 5: IMPACT OF MISSING VALUE HANDLING")
    print("=" * 80)
    
    # Check how many rows would be dropped
    all_check_cols = available_features + [c for c in required_covariates if c in df_combined.columns]
    
    print(f"\nChecking {len(all_check_cols)} columns for missing values...")
    
    # Count rows with ANY missing values
    df_check = df_combined[all_check_cols]
    complete_rows = df_check.dropna()
    
    print(f"\nRows before dropna: {len(df_combined)}")
    print(f"Rows after dropna: {len(complete_rows)}")
    print(f"Rows dropped: {len(df_combined) - len(complete_rows)} ({((len(df_combined) - len(complete_rows))/len(df_combined)*100):.1f}%)")
    
    # Show group distribution after dropping
    if len(complete_rows) > 0:
        # Merge back the group info
        complete_with_groups = df_combined.loc[complete_rows.index, ['analysis_group']].copy()
        group_counts_after = complete_with_groups['analysis_group'].value_counts().sort_index()
        
        print("\nGroup distribution AFTER dropping missing values:")
        for group_id, count in group_counts_after.items():
            group_label = config.GROUP_LABELS.get(group_id, f"Group {group_id}")
            original_count = group_counts.get(group_id, 0)
            dropped = original_count - count
            print(f"   {group_label}: {count} subjects (dropped {dropped})")
    
    # Step 6: Show which subjects are being dropped
    print("\n" + "=" * 80)
    print("STEP 6: WHERE ARE THE MISSING VALUES?")
    print("=" * 80)
    
    # Find columns with most missing values
    missing_summary = []
    for col in all_check_cols:
        missing_count = df_combined[col].isna().sum()
        if missing_count > 0:
            missing_summary.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': (missing_count / len(df_combined)) * 100
            })
    
    missing_df = pd.DataFrame(missing_summary).sort_values('missing_count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\nTop 20 columns with missing values:")
        print(missing_df.head(20).to_string(index=False))
    else:
        print("\n✓ No missing values found!")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\n📋 SUMMARY:")
    print(f"   Raw data loaded: {total_raw} subjects from {len(sheet_names)} sheets")
    print(f"   After missing value handling: {len(complete_rows)} subjects")
    print(f"   Data loss: {len(df_combined) - len(complete_rows)} subjects ({((len(df_combined) - len(complete_rows))/len(df_combined)*100):.1f}%)")
    
    print("\n💡 RECOMMENDATIONS:")
    if len(missing_df) > 0:
        high_missing = missing_df[missing_df['missing_pct'] > 30]
        if len(high_missing) > 0:
            print(f"   ⚠️  {len(high_missing)} features have >30% missing data")
            print("   → Consider removing these features from config.ALL_FITBIT_FEATURES")
    
    missing_covs = [c for c in required_covariates if c not in df_combined.columns]
    if missing_covs:
        print(f"   ⚠️  Missing covariates: {missing_covs}")
        print("   → Use add_covariates_helper.py to merge demographic data")
        print("   → Or remove from config.COVARIATES if not available")
    
    if len(complete_rows) < total_raw * 0.5:
        print(f"   ⚠️  Losing >50% of data due to missing values!")
        print("   → Consider using method='median' or 'mean' instead of 'drop'")
        print("   → Edit data_loader.py line ~280: method='median'")
    
    print("\n" + "=" * 80)
    
    return df_combined, complete_rows


if __name__ == "__main__":
    df_raw, df_clean = diagnose_data_loading()
    
    print("\n\n🔍 Want more details? Check these files:")
    print(f"   - Raw combined data: df_raw (shape: {df_raw.shape})")
    print(f"   - After cleaning: df_clean (shape: {df_clean.shape})")
    print("\nSaved diagnostic results to console output above.")