"""
Main Pipeline Script
Orchestrates the complete ADHD Digital Phenotype Analysis Pipeline
Executes all phases: Data Loading -> Residualization -> Modeling -> Visualization
"""

import logging
import sys
import argparse
from datetime import datetime
import json

# Import pipeline modules
import config
import utils
from data_loader import load_and_prepare_data
from residualization import residualize_all_features, validate_residualization
from univariate_tests import run_all_univariate_tests, create_effect_size_summary
from predictive_models import run_comparison, plot_model_results
from visualization import run_pca_analysis

# Get logger for this module
logger = logging.getLogger(__name__)


def main(tune_hyperparameters=False, skip_validation=False):
    """
    Main pipeline execution function
    
    Parameters:
    -----------
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning (takes longer)
    skip_validation : bool
        Whether to skip residualization validation
    """
    
    # Start timer
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("ADHD DIGITAL PHENOTYPE ANALYSIS PIPELINE")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        # ====================================================================
        # PHASE 0: DATA LOADING AND PREPARATION
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 0: DATA LOADING AND PREPARATION")
        logger.info("=" * 80)
        
        df, features = load_and_prepare_data(
            fitbit_path=config.FITBIT_FILE,
            metadata_path=config.METADATA_FILE
        )
        
        logger.info(f"Loaded dataset: {len(df)} subjects, {len(features)} features")
        
        # Check that we have multiple groups
        unique_groups = df['analysis_group'].nunique()
        if unique_groups < 2:
            logger.error(f"Only {unique_groups} group(s) found in data. Need at least 2 groups for analysis.")
            logger.error("Check that your Excel file has multiple sheets (e.g., NC_Controls, ADHD_Unmedicated, ADHD_Stimulants)")
            raise ValueError("Insufficient groups for analysis")
        
        logger.info(f"Found {unique_groups} groups for analysis")
        
        # Perform data quality check
        quality_report = utils.check_data_quality(df, features, 'analysis_group')
        
        # ====================================================================
        # PHASE 1: RESIDUALIZATION
        # ====================================================================
        df_residualized, residual_stats = residualize_all_features(
            df, features, control_group=config.GROUP_DEFINITIONS['control']
        )
        
        # Validate residualization
        if not skip_validation and len(features) > 0:
            validation_results = validate_residualization(
                df, df_residualized, features, covariate='interview_age'
            )
        
        # Save residualized data
        residualized_file = config.OUTPUT_DIR / "residualized_data.csv"
        df_residualized.to_csv(residualized_file, index=False)
        logger.info(f"Residualized data saved to {residualized_file}")
        
        # ====================================================================
        # PHASE 1.5: UNIVARIATE STATISTICAL TESTS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1.5: UNIVARIATE STATISTICAL TESTS")
        logger.info("=" * 80)
        
        # Get successful features
        successful_features = [f for f in features if f in df_residualized.columns]
        
        # Run univariate tests
        univariate_results = run_all_univariate_tests(df_residualized, successful_features)
        
        # Create effect size summary
        effect_size_summary = create_effect_size_summary(univariate_results)
        
        # ====================================================================
        # PHASE 2: PREDICTIVE MODELING
        # ====================================================================
        all_model_results = {}
        
        # Get successful features (features that were residualized)
        successful_features = [f for f in features if f in df_residualized.columns]
        
        logger.info(f"\nRunning predictive models with {len(successful_features)} features")
        
        if len(successful_features) == 0:
            logger.error("No successful features from residualization. Cannot run predictive models.")
        else:
            # Run all specified comparisons
            for group1_name, group2_name in config.COMPARISON_PAIRS:
                group1 = config.GROUP_DEFINITIONS[group1_name]
                group2 = config.GROUP_DEFINITIONS[group2_name]
                
                # Check if both groups exist in data
                if group1 not in df_residualized['analysis_group'].unique():
                    logger.warning(f"Group {group1_name} not found in data. Skipping comparison.")
                    continue
                if group2 not in df_residualized['analysis_group'].unique():
                    logger.warning(f"Group {group2_name} not found in data. Skipping comparison.")
                    continue
                
                try:
                    # Run comparison
                    comparison_results = run_comparison(
                        df_residualized,
                        successful_features,
                        group1,
                        group2,
                        tune_hyperparameters=tune_hyperparameters
                    )
                    
                    # Store results
                    comparison_name = comparison_results['comparison']
                    all_model_results[comparison_name] = comparison_results
                    
                    # Create visualizations
                    plot_model_results(comparison_results, 'model')
                    
                    # Save feature importance/coefficients
                    lr_coef = comparison_results['logistic_regression']['coefficients']
                    rf_imp = comparison_results['random_forest']['feature_importances']
                    
                    lr_coef.to_csv(config.OUTPUT_DIR / f"lr_coefficients_{comparison_name}.csv", index=False)
                    rf_imp.to_csv(config.OUTPUT_DIR / f"rf_importances_{comparison_name}.csv", index=False)
                    
                    logger.info(f"Results saved for comparison: {comparison_name}")
                    
                except Exception as e:
                    logger.error(f"Error in comparison {group1_name} vs {group2_name}: {str(e)}", exc_info=True)
                    continue
        
        # ====================================================================
        # PHASE 3: PCA VISUALIZATION
        # ====================================================================
        pca_results = {}
        if len(successful_features) > 0:
            try:
                pca_results = run_pca_analysis(df_residualized, successful_features)
            except Exception as e:
                logger.error(f"Error in PCA analysis: {str(e)}", exc_info=True)
                pca_results = {
                    'explained_variance': [],
                    'separation_statistics': {}
                }
        else:
            logger.error("No successful features. Skipping PCA analysis.")
        
        # ====================================================================
        # GENERATE SUMMARY REPORT
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("=" * 80)
        
        summary_results = {
            'data_quality': quality_report,
            'univariate_tests': univariate_results,
            'effect_size_summary': effect_size_summary.to_dict() if not effect_size_summary.empty else {},
            'model_results': all_model_results,
            'pca_results': {
                'explained_variance': pca_results.get('explained_variance', []).tolist() if len(pca_results.get('explained_variance', [])) > 0 else [],
                'separation_statistics': pca_results.get('separation_statistics', {})
            }
        }
        
        # Create text summary
        utils.create_summary_report(summary_results, 'summary_report.txt')
        
        # Save JSON results
        utils.save_results(summary_results, 'pipeline_results.json')
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Results saved to: {config.OUTPUT_DIR}")
        logger.info("=" * 80)
        
        # Print key findings
        print("\n" + "=" * 80)
        print("KEY FINDINGS SUMMARY")
        print("=" * 80)
        print(f"\nDataset: {len(df)} subjects across {len(config.GROUP_LABELS)} groups")
        print(f"Features analyzed: {len(successful_features)}")
        print(f"\nGroup Distribution (After cleaning):")
        if 'group_counts' in quality_report:
            for group_id, count in quality_report['group_counts'].items():
                group_name = config.GROUP_LABELS.get(int(group_id), f"Group {group_id}")
                print(f"  {group_name}: {count}")
        
        print(f"\nModel Performance:")
        if all_model_results:
            for comparison_name, results in all_model_results.items():
                print(f"\n  {comparison_name}:")
                print(f"    Logistic Regression ROC-AUC: {results['logistic_regression']['roc_auc']:.4f}")
                print(f"    Random Forest ROC-AUC: {results['random_forest']['roc_auc']:.4f}")
        else:
            print("  No model results generated.")
            
        print(f"\nPCA Explained Variance:")
        if pca_results and 'explained_variance' in pca_results and len(pca_results['explained_variance']) > 0:
            for i, var in enumerate(pca_results['explained_variance'][:3], 1):
                print(f"  PC{i}: {var*100:.2f}%")
        else:
            print("  PCA analysis not completed")
        
        print("\n" + "=" * 80)
        print(f"All outputs saved to: {config.OUTPUT_DIR}")
        print("=" * 80)
        
        return summary_results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ADHD Digital Phenotype Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main_pipeline.py
  
  # Run with hyperparameter tuning (slower but potentially better results)
  python main_pipeline.py --tune
  
  # Skip residualization validation (faster)
  python main_pipeline.py --skip-validation
        """
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning for models (takes longer)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip residualization validation step'
    )
    
    parser.add_argument(
        '--fitbit-file',
        type=str,
        help='Path to Fitbit data file (overrides config)'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=str,
        help='Path to metadata file (overrides config)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # === FIX: CONFIGURE LOGGING HERE ===
    # Setup logging (only in the main entry point)
    # Use 'w' mode to overwrite the log file for each new run
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # ===================================
    
    # Override config if files specified
    if args.fitbit_file:
        config.FITBIT_FILE = args.fitbit_file
    if args.metadata_file:
        config.METADATA_FILE = args.metadata_file
    
    # Run pipeline
    try:
        results = main(
            tune_hyperparameters=args.tune,
            skip_validation=args.skip_validation
        )
        sys.exit(0)
    except Exception as e:
        # Logger is now configured, so this will work
        logger.critical("PIPELINE FAILED WITH UNRECOVERABLE ERROR", exc_info=True)
        sys.exit(1)