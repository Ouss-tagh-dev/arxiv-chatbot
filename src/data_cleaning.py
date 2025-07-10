"""
Enhanced script to clean and prepare the arXiv data with advanced NLP preprocessing.
Run this before training your NLP model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import ArxivDataLoader
from cleaner import EnhancedArxivDataCleaner, EnhancedTextPreprocessor, analyze_dataset_quality
import pandas as pd
import logging
import argparse
from pathlib import Path
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for data processing."""
    directories = [
        "data/raw",
        "data/processed",
        "data/cleaned",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate the input data format and quality.
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if data is valid, False otherwise
    """
    required_columns = ['id', 'title', 'summary', 'published', 'primary_category']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for minimum data quality
    if len(df) < 10:
        logger.error("Dataset too small (less than 10 articles)")
        return False
    
    # Check for basic data integrity
    empty_titles = df['title'].isna().sum()
    empty_summaries = df['summary'].isna().sum()
    
    if empty_titles > len(df) * 0.5:
        logger.warning(f"High percentage of empty titles: {empty_titles/len(df)*100:.1f}%")
    
    if empty_summaries > len(df) * 0.5:
        logger.warning(f"High percentage of empty summaries: {empty_summaries/len(df)*100:.1f}%")
    
    logger.info("Input data validation passed")
    return True

def create_sample_dataset(df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    Create a sample dataset for testing.
    
    Args:
        df: Full DataFrame
        sample_size: Size of sample to create
        
    Returns:
        Sample DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    # Stratified sampling by primary category
    sample_df = df.groupby('primary_category').apply(
        lambda x: x.sample(min(len(x), max(1, sample_size // df['primary_category'].nunique())))
    ).reset_index(drop=True)
    
    logger.info(f"Created sample dataset with {len(sample_df)} articles")
    return sample_df

def main():
    """Enhanced data cleaning pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced ArXiv Data Cleaning')
    parser.add_argument('--input', default='data/raw/articles.csv', 
                       help='Path to input CSV file')
    parser.add_argument('--output', default='data/processed/articles_enhanced_clean.csv',
                       help='Path to output CSV file')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Create sample dataset of specified size')
    parser.add_argument('--deep-clean', action='store_true',
                       help='Perform deep NLP cleaning (slower but more thorough)')
    parser.add_argument('--use-spacy', action='store_true',
                       help='Use spaCy for advanced NLP processing')
    parser.add_argument('--remove-stopwords', action='store_true',
                       help='Remove stop words from text')
    parser.add_argument('--lemmatize', action='store_true',
                       help='Lemmatize words')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate cleaned data without processing')
    
    args = parser.parse_args()
    
    try:
        # Setup directories
        setup_directories()
        
        # Load raw data
        logger.info(f"Loading raw data from {args.input}...")
        start_time = time.time()
        
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        
        loader = ArxivDataLoader(args.input)
        df = loader.load_data()
        
        logger.info(f"Loaded {len(df)} articles in {time.time() - start_time:.2f} seconds")
        
        # Validate input data
        if not validate_input_data(df):
            logger.error("Input data validation failed")
            return
        
        # Create sample if requested
        if args.sample_size:
            df = create_sample_dataset(df, args.sample_size)
        
        # If only validation requested, load cleaned data and validate
        if args.validate_only:
            if os.path.exists(args.output):
                logger.info("Validating existing cleaned data...")
                df_clean = pd.read_csv(args.output)
                quality_metrics = analyze_dataset_quality(df_clean)
                
                print("\nDataset Quality Analysis:")
                for key, value in quality_metrics.items():
                    print(f"  {key}: {value}")
                
                return
            else:
                logger.error("No cleaned data found to validate")
                return
        
        # Initialize enhanced cleaner
        logger.info("Initializing enhanced data cleaner...")
        cleaner = EnhancedArxivDataCleaner(use_spacy=args.use_spacy)
        
        # Clean the data
        logger.info("Starting enhanced data cleaning...")
        start_time = time.time()
        
        df_clean = cleaner.clean_dataset(df, deep_clean=args.deep_clean)
        
        cleaning_time = time.time() - start_time
        logger.info(f"Enhanced cleaning completed in {cleaning_time:.2f} seconds")
        
        # Additional text preprocessing if requested
        if args.remove_stopwords or args.lemmatize:
            logger.info("Performing additional text preprocessing...")
            preprocessor = EnhancedTextPreprocessor(
                remove_stopwords=args.remove_stopwords,
                lemmatize=args.lemmatize
            )
            
            # Process titles and summaries
            if 'title_clean' in df_clean.columns:
                df_clean['title_processed'] = df_clean['title_clean'].apply(
                    preprocessor.preprocess_text
                )
            
            if 'summary_clean' in df_clean.columns:
                df_clean['summary_processed'] = df_clean['summary_clean'].apply(
                    preprocessor.preprocess_text
                )
        
        # Save cleaned data
        logger.info("Saving cleaned data...")
        cleaner.save_cleaned_data(df_clean, args.output, save_separate_fields=True)
        
        # Generate and save comprehensive statistics
        stats = cleaner.get_cleaning_stats()
        quality_metrics = analyze_dataset_quality(df_clean)
        
        # Combine all statistics
        comprehensive_stats = {
            'processing_info': {
                'input_file': args.input,
                'output_file': args.output,
                'processing_time_seconds': cleaning_time,
                'deep_clean': args.deep_clean,
                'use_spacy': args.use_spacy,
                'remove_stopwords': args.remove_stopwords,
                'lemmatize': args.lemmatize,
                'sample_size': args.sample_size
            },
            'cleaning_statistics': stats,
            'quality_metrics': quality_metrics
        }
        
        # Save comprehensive statistics
        stats_path = Path(args.output).parent / "comprehensive_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCED DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Input articles: {stats.get('initial_count', 'N/A')}")
        print(f"Final articles: {stats.get('final_count', 'N/A')}")
        print(f"Removed articles: {stats.get('removed_count', 'N/A')}")
        print(f"Removal rate: {stats.get('removal_rate', 0)*100:.1f}%")
        print(f"Processing time: {cleaning_time:.2f} seconds")
        
        print("\nCleaning Statistics:")
        for key, value in stats.items():
            if key not in ['initial_count', 'final_count', 'removed_count', 'removal_rate']:
                print(f"  {key}: {value}")
        
        print("\nQuality Metrics:")
        for key, value in quality_metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nFiles saved:")
        print(f"  Main dataset: {args.output}")
        print(f"  Essential fields: {Path(args.output).parent / f'essential_{Path(args.output).name}'}")
        print(f"  Text-only: {Path(args.output).parent / f'text_only_{Path(args.output).name}'}")
        print(f"  Statistics: {stats_path}")
        
        # Data quality recommendations
        print("\nData Quality Recommendations:")
        
        if quality_metrics.get('avg_title_length', 0) < 5:
            print("  ⚠️  Average title length is very short - consider reviewing title extraction")
        
        if quality_metrics.get('avg_summary_length', 0) < 50:
            print("  ⚠️  Average summary length is short - consider reviewing summary extraction")
        
        if len(df_clean) < len(df) * 0.7:
            print("  ⚠️  High removal rate - consider relaxing cleaning criteria")
        
        if quality_metrics.get('unique_categories', 0) < 10:
            print("  ⚠️  Low category diversity - consider checking category extraction")
        
        print("\n✅ Enhanced data cleaning completed successfully!")
        
    except FileNotFoundError:
        logger.error(f"Raw data file not found at '{args.input}'")
        logger.info("Please ensure you have downloaded the arXiv data first.")
    except Exception as e:
        logger.error(f"Error during enhanced data cleaning: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()