"""
Helper script to add covariates to Fitbit data
If your Fitbit sheets don't have covariates (age, sex, income, race),
this script can help merge them from ABCD core demographic files
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_covariates_to_fitbit(fitbit_file: str,
                             demo_file: str,
                             output_file: str = None):
    """
    Merge demographic covariates into Fitbit data
    
    Parameters:
    -----------
    fitbit_file : str
        Path to Fitbit Excel file (multi-sheet)
    demo_file : str
        Path to ABCD demographics file with covariates
    output_file : str, optional
        Output file path. If None, creates fitbit_data_with_covariates.xlsx
    """
    
    logger.info(f"Loading Fitbit data from {fitbit_file}")
    
    # Load demographics file
    logger.info(f"Loading demographics from {demo_file}")
    demo_df = pd.read_excel(demo_file) if demo_file.endswith('.xlsx') else pd.read_csv(demo_file)
    
    logger.info(f"Demographics file has {len(demo_df)} records")
    logger.info(f"Demographics columns: {list(demo_df.columns)}")
    
    # Required covariates to look for
    required_covariates = ['interview_age', 'sex', 'family_income', 'race_eth_cat']
    
    # Check which covariates are available
    available_covariates = []
    for cov in required_covariates:
        if cov in demo_df.columns:
            available_covariates.append(cov)
        else:
            logger.warning(f"Covariate '{cov}' not found in demographics file")
    
    if not available_covariates:
        logger.error("No required covariates found in demographics file")
        return
    
    logger.info(f"Found covariates: {available_covariates}")
    
    # Select relevant columns from demographics
    merge_cols = ['subjectkey'] + available_covariates
    demo_subset = demo_df[merge_cols].drop_duplicates(subset=['subjectkey'])
    
    # Load and process each sheet from Fitbit file
    excel_file = pd.ExcelFile(fitbit_file)
    sheet_names = excel_file.sheet_names
    
    logger.info(f"Processing {len(sheet_names)} sheets from Fitbit file")
    
    # Create output file
    if output_file is None:
        output_file = fitbit_file.replace('.xlsx', '_with_covariates.xlsx')
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name in sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")
            
            # Load sheet
            df_sheet = pd.read_excel(fitbit_file, sheet_name=sheet_name)
            
            # Check if covariates already exist
            existing_covs = [c for c in available_covariates if c in df_sheet.columns]
            if existing_covs:
                logger.info(f"  Sheet already has covariates: {existing_covs}")
                df_merged = df_sheet
            else:
                # Merge with demographics
                df_merged = df_sheet.merge(demo_subset, on='subjectkey', how='left')
                
                # Check merge success
                merged_count = df_merged[available_covariates[0]].notna().sum()
                logger.info(f"  Merged covariates for {merged_count}/{len(df_merged)} subjects")
            
            # Write to output
            df_merged.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"  Written {len(df_merged)} rows to sheet '{sheet_name}'")
    
    logger.info(f"\nOutput saved to: {output_file}")
    logger.info("You can now use this file in your analysis pipeline")


def inspect_data_structure(fitbit_file: str, demo_file: str = None):
    """
    Inspect the structure of your data files to help with merging
    
    Parameters:
    -----------
    fitbit_file : str
        Path to Fitbit Excel file
    demo_file : str, optional
        Path to demographics file
    """
    print("\n" + "=" * 80)
    print("DATA STRUCTURE INSPECTION")
    print("=" * 80)
    
    # Inspect Fitbit file
    print(f"\nFitbit File: {fitbit_file}")
    print("-" * 80)
    
    excel_file = pd.ExcelFile(fitbit_file)
    sheet_names = excel_file.sheet_names
    
    print(f"Number of sheets: {len(sheet_names)}")
    print(f"Sheet names: {sheet_names}\n")
    
    for sheet_name in sheet_names:
        df = pd.read_excel(fitbit_file, sheet_name=sheet_name)
        print(f"Sheet '{sheet_name}':")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Check for subject key
        if 'subjectkey' in df.columns:
            print(f"  ✓ Has 'subjectkey' column")
            print(f"  Unique subjects: {df['subjectkey'].nunique()}")
        else:
            print(f"  ✗ Missing 'subjectkey' column")
            print(f"  Available ID columns: {[c for c in df.columns if 'subject' in c.lower() or 'id' in c.lower()]}")
        
        # Check for covariates
        covariates = ['interview_age', 'sex', 'family_income', 'race_eth_cat']
        has_covs = [c for c in covariates if c in df.columns]
        missing_covs = [c for c in covariates if c not in df.columns]
        
        if has_covs:
            print(f"  ✓ Has covariates: {has_covs}")
        if missing_covs:
            print(f"  ✗ Missing covariates: {missing_covs}")
        
        print()
    
    # Inspect demographics file if provided
    if demo_file:
        print(f"\nDemographics File: {demo_file}")
        print("-" * 80)
        
        demo_df = pd.read_excel(demo_file) if demo_file.endswith('.xlsx') else pd.read_csv(demo_file)
        print(f"Rows: {len(demo_df)}")
        print(f"Columns: {len(demo_df.columns)}")
        print(f"\nKey columns:")
        
        # Check for subject key
        if 'subjectkey' in demo_df.columns:
            print(f"  ✓ Has 'subjectkey' column")
            print(f"  Unique subjects: {demo_df['subjectkey'].nunique()}")
        else:
            print(f"  ✗ Missing 'subjectkey' column")
            print(f"  Available ID columns: {[c for c in demo_df.columns if 'subject' in c.lower() or 'id' in c.lower()]}")
        
        # Check for covariates
        has_covs = [c for c in covariates if c in demo_df.columns]
        missing_covs = [c for c in covariates if c not in demo_df.columns]
        
        if has_covs:
            print(f"  ✓ Has covariates: {has_covs}")
        if missing_covs:
            print(f"  ✗ Missing covariates: {missing_covs}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Inspect data structure:")
        print("    python add_covariates_helper.py inspect <fitbit_file> [demo_file]")
        print()
        print("  Add covariates:")
        print("    python add_covariates_helper.py merge <fitbit_file> <demo_file> [output_file]")
        print()
        print("Example:")
        print("    python add_covariates_helper.py inspect fitbit_data.xlsx demographics.csv")
        print("    python add_covariates_helper.py merge fitbit_data.xlsx demographics.csv")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'inspect':
        fitbit_file = sys.argv[2]
        demo_file = sys.argv[3] if len(sys.argv) > 3 else None
        inspect_data_structure(fitbit_file, demo_file)
    
    elif command == 'merge':
        if len(sys.argv) < 4:
            print("Error: Need both fitbit_file and demo_file for merge")
            sys.exit(1)
        
        fitbit_file = sys.argv[2]
        demo_file = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        
        add_covariates_to_fitbit(fitbit_file, demo_file, output_file)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'inspect' or 'merge'")
        sys.exit(1)