"""
Data cleaning and preprocessing module for ArXiv chatbot project.
Handles text cleaning, normalization, and data quality improvements.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import unicodedata
from datetime import datetime
import pytz  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivDataCleaner:
    """
    Comprehensive data cleaning for ArXiv articles.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.cleaning_stats = {}
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline for the dataset.
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Track initial state
        initial_count = len(df_clean)
        self.cleaning_stats['initial_count'] = initial_count
        
        # 1. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # 2. Clean text fields
        df_clean = self._clean_text_fields(df_clean)
        
        # 3. Normalize authors
        df_clean = self._normalize_authors(df_clean)
        
        # 4. Clean categories
        df_clean = self._clean_categories(df_clean)
        
        # 5. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 6. Validate dates
        df_clean = self._validate_dates(df_clean)
        
        # 7. Clean DOIs
        df_clean = self._clean_dois(df_clean)
        
        # 8. Filter out invalid entries
        df_clean = self._filter_invalid_entries(df_clean)
        
        # Final statistics
        final_count = len(df_clean)
        self.cleaning_stats['final_count'] = final_count
        self.cleaning_stats['removed_count'] = initial_count - final_count
        self.cleaning_stats['removal_rate'] = (initial_count - final_count) / initial_count
        
        logger.info(f"Cleaning completed. Removed {initial_count - final_count} articles ({self.cleaning_stats['removal_rate']:.2%})")
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles based on ArXiv ID."""
        initial_count = len(df)
        
        # Remove duplicates based on ArXiv ID
        df_clean = df.drop_duplicates(subset=['id'], keep='last')
        
        # Also check for near-duplicates based on title similarity
        df_clean = self._remove_title_duplicates(df_clean)
        
        removed_count = initial_count - len(df_clean)
        self.cleaning_stats['duplicates_removed'] = removed_count
        
        logger.info(f"Removed {removed_count} duplicate articles")
        return df_clean
    
    def _remove_title_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with very similar titles."""
        from difflib import SequenceMatcher
        
        # Simple approach: remove articles with identical titles after normalization
        df['title_normalized'] = df['title'].str.lower().str.strip()
        df_clean = df.drop_duplicates(subset=['title_normalized'], keep='first')
        df_clean = df_clean.drop('title_normalized', axis=1)
        
        return df_clean
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean title and summary text fields."""
        logger.info("Cleaning text fields...")
        
        # Clean titles
        df['title'] = df['title'].apply(self._clean_text)
        
        # Clean summaries
        df['summary'] = df['summary'].apply(self._clean_text)
        
        # Clean journal references
        df['journal_ref'] = df['journal_ref'].apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Clean individual text strings.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove newlines and replace with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove LaTeX commands (basic cleanup)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-.,;:!?()[\]{}"\']', ' ', text)
        
        # Clean up multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize author names and affiliations."""
        logger.info("Normalizing author names...")
        
        def normalize_author_string(author_string):
            if pd.isna(author_string):
                return ""
            
            # Split authors by semicolon
            authors = [author.strip() for author in str(author_string).split(';')]
            
            # Clean each author name
            cleaned_authors = []
            for author in authors:
                # Remove extra whitespace
                author = re.sub(r'\s+', ' ', author).strip()
                
                # Remove common prefixes/suffixes
                author = re.sub(r'^(Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+', '', author, flags=re.IGNORECASE)
                
                if author:  # Only add non-empty authors
                    cleaned_authors.append(author)
            
            return '; '.join(cleaned_authors)
        
        df['author'] = df['author'].apply(normalize_author_string)
        
        return df
    
    def _clean_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize category information."""
        logger.info("Cleaning categories...")
        
        # Clean primary category
        df['primary_category'] = df['primary_category'].str.strip()
        
        # Clean category list
        def clean_category_string(cat_string):
            if pd.isna(cat_string):
                return ""
            
            categories = [cat.strip() for cat in str(cat_string).split(';')]
            cleaned_categories = [cat for cat in categories if cat]
            
            return '; '.join(cleaned_categories)
        
        df['category'] = df['category'].apply(clean_category_string)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Count missing values
        missing_stats = df.isnull().sum()
        self.cleaning_stats['missing_values'] = missing_stats.to_dict()
        
        # Fill missing values with appropriate defaults
        df['title'] = df['title'].fillna('No Title')
        df['summary'] = df['summary'].fillna('No Summary Available')
        df['author'] = df['author'].fillna('Unknown Author')
        df['primary_category'] = df['primary_category'].fillna('Unknown')
        df['category'] = df['category'].fillna('Unknown')
        
        # Keep DOI and journal_ref as nullable
        
        return df
    
    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean date fields."""
        logger.info("Validating dates...")
        
        # Parse dates
        df['published_date'] = pd.to_datetime(df['published'], errors='coerce')
        df['updated_date'] = pd.to_datetime(df['updated'], errors='coerce')
        
        # Count invalid dates
        invalid_published = df['published_date'].isna().sum()
        invalid_updated = df['updated_date'].isna().sum()
        
        self.cleaning_stats['invalid_published_dates'] = invalid_published
        self.cleaning_stats['invalid_updated_dates'] = invalid_updated
        
        # Filter out articles with invalid published dates
        df = df[df['published_date'].notna()]
        
        return df
    
    def _clean_dois(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate DOI fields."""
        logger.info("Cleaning DOIs...")
        
        def clean_doi(doi):
            if pd.isna(doi):
                return None
            
            doi = str(doi).strip()
            
            # Extract DOI from URL if present
            if 'doi.org' in doi:
                doi = doi.split('doi.org/')[-1]
            
            # Basic DOI format validation
            if re.match(r'^10\.\d+/', doi):
                return doi
            
            return None
        
        df['doi'] = df['doi'].apply(clean_doi)
        
        return df
    
    def _filter_invalid_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid or low-quality entries."""
        logger.info("Filtering invalid entries...")
        
        initial_count = len(df)
        
        # Remove articles with very short titles (likely corrupted)
        df = df[df['title'].str.len() >= 10]
        
        # Remove articles with very short summaries (likely incomplete)
        df = df[df['summary'].str.len() >= 50]
        
        # Remove articles with invalid ArXiv IDs
        df = df[df['id'].str.match(r'^\d+\.\d+', na=False)]
        
        # Remove articles with future dates (likely errors)
        # FIX: Make current_date timezone-aware to match published_date
        current_date = datetime.now(pytz.UTC)  # Create timezone-aware datetime
        df = df[df['published_date'] <= current_date]
        
        filtered_count = initial_count - len(df)
        self.cleaning_stats['filtered_invalid'] = filtered_count
        
        logger.info(f"Filtered out {filtered_count} invalid entries")
        
        return df
    
    def get_cleaning_stats(self) -> Dict:
        """Get statistics about the cleaning process."""
        return self.cleaning_stats
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """
        Save cleaned data to file.
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save the cleaned data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        
        # Save cleaning statistics
        stats_path = output_path.parent / "cleaning_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.cleaning_stats, f, indent=2, default=str)
        
        logger.info(f"Saved cleaned data to {output_path}")
        logger.info(f"Saved cleaning statistics to {stats_path}")


class TextPreprocessor:
    """
    Advanced text preprocessing for semantic search.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> set:
        """Load stop words for text preprocessing."""
        # Basic English stop words
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i',
            'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now'
        }
    
    def preprocess_for_search(self, text: str) -> str:
        """
        Preprocess text for semantic search.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length
            
        Returns:
            List of keywords
        """
        if pd.isna(text):
            return []
        
        # Preprocess text
        text = self.preprocess_for_search(text)
        
        # Split into words
        words = text.split()
        
        # Filter words
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in self.stop_words
        ]
        
        return keywords


if __name__ == "__main__":
    # Example usage
    from data_loader import ArxivDataLoader
    
    # Load sample data
    loader = ArxivDataLoader("data/raw/articles.csv")
    df = loader.load_data(nrows=1000)
    
    # Clean data
    cleaner = ArxivDataCleaner()
    df_clean = cleaner.clean_dataset(df)
    
    # Print cleaning statistics
    print("Cleaning Statistics:")
    print(cleaner.get_cleaning_stats())
    
    # Save cleaned data
    cleaner.save_cleaned_data(df_clean, "data/processed/articles_clean.csv")