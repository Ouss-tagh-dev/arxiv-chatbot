"""
Data loading and management module for ArXiv chatbot project.
Handles loading, validating, and basic processing of ArXiv data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for parsed data
_DATA_CACHE = {}

class ArxivDataLoader:
    """
    Handles loading and basic processing of ArXiv data from CSV files.
    Optimized with caching to avoid re-parsing data.
    """
    
    def __init__(self, data_path: str = "data/raw/articles.csv"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the main articles CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.metadata = {}
        
    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load ArXiv data from CSV file with caching.
        
        Args:
            nrows: Number of rows to load (None for all)
            
        Returns:
            DataFrame with loaded data
        """
        # Create cache key
        cache_key = f"{self.data_path}_{nrows}"
        
        # Check if data is already cached
        if cache_key in _DATA_CACHE:
            logger.info(f"Using cached data for {self.data_path}")
            self.df = _DATA_CACHE[cache_key]
            self._compute_metadata()
            return self.df
            
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Check if file exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Optimized data loading with better dtypes
            self.df = pd.read_csv(
                self.data_path,
                nrows=nrows,
                dtype={
                    'id': 'string',
                    'title': 'string',
                    'author': 'string',
                    'summary': 'string',
                    'primary_category': 'string',
                    'category': 'string',
                    'doi': 'string',
                    'journal_ref': 'string',
                    'published': 'string',  # Parse dates manually for better performance
                    'updated': 'string'
                },
                # Use chunking for large files
                chunksize=100000 if nrows is None else None
            )
            
            # If chunksize was used, concatenate chunks
            if isinstance(self.df, pd.io.parsers.TextFileReader):
                chunks = []
                for chunk in self.df:
                    chunks.append(chunk)
                self.df = pd.concat(chunks, ignore_index=True)
            
            # Parse dates efficiently
            self._parse_dates_efficiently()
            
            # Cache the processed data
            _DATA_CACHE[cache_key] = self.df.copy()
            
            logger.info(f"Loaded and cached {len(self.df)} articles")
            self._compute_metadata()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _parse_dates_efficiently(self):
        """Parse dates efficiently using vectorized operations."""
        if self.df is None:
            return
            
        # Parse dates only if not already parsed
        if 'published_date' not in self.df.columns:
            self.df['published_date'] = pd.to_datetime(self.df['published'], errors='coerce')
        if 'updated_date' not in self.df.columns:
            self.df['updated_date'] = pd.to_datetime(self.df['updated'], errors='coerce')
    
    def _compute_metadata(self):
        """Compute basic metadata about the dataset."""
        if self.df is None:
            return
        
        # Compute metadata efficiently
        self.metadata = {
            'total_articles': len(self.df),
            'unique_authors': len(self.df['author'].dropna().unique()),
            'date_range': {
                'start': self.df['published_date'].min(),
                'end': self.df['published_date'].max()
            },
            'categories': self.df['primary_category'].value_counts().to_dict(),
            'missing_summaries': self.df['summary'].isna().sum(),
            'missing_titles': self.df['title'].isna().sum()
        }
    
    def get_metadata(self) -> Dict:
        """Get dataset metadata."""
        return self.metadata
    
    def filter_by_category(self, categories: List[str]) -> pd.DataFrame:
        """
        Filter articles by primary category.
        
        Args:
            categories: List of categories to include
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        mask = self.df['primary_category'].isin(categories)
        return self.df[mask].copy()
    
    def filter_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter articles by publication date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        mask = (self.df['published_date'] >= start) & (self.df['published_date'] <= end)
        return self.df[mask].copy()
    
    def get_sample(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Get a random sample of articles.
        
        Args:
            n: Number of articles to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Sample DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.sample(n=min(n, len(self.df)), random_state=random_state)
    
    def get_articles_by_author(self, author_pattern: str) -> pd.DataFrame:
        """
        Get articles by author name pattern.
        
        Args:
            author_pattern: Pattern to search in author names
            
        Returns:
            DataFrame with matching articles
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        mask = self.df['author'].str.contains(author_pattern, case=False, na=False)
        return self.df[mask].copy()
    
    def get_top_categories(self, n: int = 10) -> pd.Series:
        """
        Get top N categories by article count.
        
        Args:
            n: Number of top categories to return
            
        Returns:
            Series with category counts
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df['primary_category'].value_counts().head(n)
    
    def search_by_keywords(self, keywords: List[str], field: str = 'summary') -> pd.DataFrame:
        """
        Search articles by keywords in specified field.
        
        Args:
            keywords: List of keywords to search for
            field: Field to search in ('title', 'summary', 'category')
            
        Returns:
            DataFrame with matching articles
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if field not in ['title', 'summary', 'category']:
            raise ValueError("Field must be 'title', 'summary', or 'category'")
        
        # Create a combined mask for all keywords
        mask = pd.Series([True] * len(self.df))
        for keyword in keywords:
            mask &= self.df[field].str.contains(keyword, case=False, na=False)
        
        return self.df[mask].copy()
    
    def export_filtered_data(self, df: pd.DataFrame, output_path: str):
        """
        Export filtered data to CSV.
        
        Args:
            df: DataFrame to export
            output_path: Path to save the CSV file
        """
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} articles to {output_path}")


class DataSplitter:
    """
    Utility class for splitting data into train/validation/test sets.
    """
    
    @staticmethod
    def split_by_date(df: pd.DataFrame, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by date (chronological split).
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Sort by date
        df_sorted = df.sort_values('published_date')
        
        # Calculate split points
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split the data
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        return train_df, val_df, test_df
    
    @staticmethod
    def split_random(df: pd.DataFrame, 
                    train_ratio: float = 0.7, 
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Calculate split points
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:]
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    loader = ArxivDataLoader("data/raw/articles.csv")
    
    # Load sample data
    df = loader.load_data(nrows=10000)
    
    # Print metadata
    print("Dataset metadata:")
    print(loader.get_metadata())
    
    # Get top categories
    print("\nTop categories:")
    print(loader.get_top_categories())
    
    # Search example
    ml_articles = loader.search_by_keywords(['machine learning', 'neural network'], 'summary')
    print(f"\nFound {len(ml_articles)} articles about machine learning")