"""
Script to clean and prepare the arXiv data.
Run this before using the interface.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import ArxivDataLoader
from cleaner import ArxivDataCleaner
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Clean the arXiv data and prepare it for use."""
    try:
        # Load raw data
        logger.info("Loading raw data...")
        loader = ArxivDataLoader("data/raw/articles.csv")
        df = loader.load_data()
        
        logger.info(f"Loaded {len(df)} articles")
        
        # Clean the data
        logger.info("Starting data cleaning...")
        cleaner = ArxivDataCleaner()
        df_clean = cleaner.clean_dataset(df)
        
        # Save cleaned data
        logger.info("Saving cleaned data...")
        cleaner.save_cleaned_data(df_clean, "data/processed/articles_clean.csv")
        
        # Print cleaning statistics
        stats = cleaner.get_cleaning_stats()
        logger.info("Cleaning Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Verify the cleaned data
        logger.info("Verifying cleaned data...")
        df_verify = pd.read_csv("data/processed/articles_clean.csv")
        
        # Check for NaN values in critical columns
        critical_columns = ['published_date', 'primary_category', 'title', 'summary']
        for col in critical_columns:
            if col in df_verify.columns:
                nan_count = df_verify[col].isna().sum()
                logger.info(f"  {col}: {nan_count} NaN values")
        
        # Parse dates to check for issues
        df_verify['published_date'] = pd.to_datetime(df_verify['published_date'], errors='coerce')
        invalid_dates = df_verify['published_date'].isna().sum()
        logger.info(f"  Invalid dates: {invalid_dates}")
        
        if invalid_dates > 0:
            logger.warning("Found invalid dates. Cleaning further...")
            df_verify = df_verify.dropna(subset=['published_date'])
            df_verify.to_csv("data/processed/articles_clean.csv", index=False)
            logger.info(f"Saved {len(df_verify)} articles after removing invalid dates")
        
        logger.info("Data cleaning completed successfully!")
        
    except FileNotFoundError:
        logger.error("Raw data file not found at 'data/raw/articles.csv'")
        logger.info("Please ensure you have downloaded the arXiv data first.")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

if __name__ == "__main__":
    main()