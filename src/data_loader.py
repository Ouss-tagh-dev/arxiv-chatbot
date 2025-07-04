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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivDataLoader:
    """
    Handles loading and basic processing of ArXiv data from CSV files.
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
        Load ArXiv data from CSV file.
        
        Args:
            nrows: Number of rows to load (None for all)
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Check if file exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load data with proper dtype handling
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
                    'journal_ref': 'string'
                },
                low_memory=False
            )
            
            logger.info(f"Loaded {len(self.df)} articles")
            self._compute_metadata()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _compute_metadata(self):
        """Compute basic metadata about the dataset."""
        if self.df is None:
            return
        
        # Parse dates
        self.df['published_date'] = pd.to_datetime(self.df['published'], errors='coerce')
        self.df['updated_date'] = pd.to_datetime(self.df['updated'], errors='coerce')
        
        # Compute metadata
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
        
        # Create search pattern
        pattern = '|'.join(keywords)
        mask = self.df[field].str.contains(pattern, case=False, na=False)
        
        return self.df[mask].copy()
    
    def export_filtered_data(self, df: pd.DataFrame, output_path: str):
        """
        Export filtered data to CSV.
        
        Args:
            df: DataFrame to export
            output_path: Path for output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        Split data chronologically by publication date.
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Sort by publication date
        df_sorted = df.sort_values('published_date')
        
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
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
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Shuffle data
        df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_shuffled.iloc[:train_end].copy()
        val_df = df_shuffled.iloc[train_end:val_end].copy()
        test_df = df_shuffled.iloc[val_end:].copy()
        
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