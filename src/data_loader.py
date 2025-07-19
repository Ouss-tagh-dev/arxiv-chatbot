"""
Fixed data loader to handle CSV type inference issues.
Add this fix to your data_loader.py file.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

class ArxivDataLoader:
    """Fixed ArXiv data loader with proper type handling."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV with explicit data types to avoid inference issues."""
        
        # Define explicit data types for ArXiv dataset columns
        # This prevents pandas from incorrectly inferring types
        dtype_dict = {
            'id': 'str',
            'title': 'str',
            'summary': 'str',
            'author': 'str',
            'published': 'str',  # Keep as string, parse later
            'updated': 'str',    # Keep as string, parse later
            'primary_category': 'str',
            'category': 'str',
            'comment': 'str',
            'doi': 'str',
            'journal_ref': 'str'
        }
        
        chunks = []
        chunk_size = 100000  # Process in chunks to avoid memory issues
        
        try:
            # Read CSV in chunks with explicit dtypes
            for chunk in pd.read_csv(
                file_path,
                chunksize=chunk_size,
                dtype=dtype_dict,  # Explicit types prevent inference errors
                low_memory=False,  # Don't use low_memory to avoid mixed types
                na_values=['', 'nan', 'NaN', 'null', 'NULL', 'None'],
                keep_default_na=True,
                encoding='utf-8',
                on_bad_lines='skip'  # Skip problematic lines
            ):
                # Process dates properly after loading
                chunk = self._process_datetime_columns(chunk)
                chunks.append(chunk)
                
                print(f"Loaded chunk with {len(chunk)} rows")
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Fallback: try without explicit dtypes
            try:
                print("Attempting fallback loading without type inference...")
                chunks = []
                for chunk in pd.read_csv(
                    file_path,
                    chunksize=chunk_size,
                    dtype=str,  # Load everything as string first
                    low_memory=False,
                    na_values=['', 'nan', 'NaN', 'null', 'NULL', 'None'],
                    keep_default_na=True,
                    encoding='utf-8',
                    on_bad_lines='skip'
                ):
                    chunk = self._process_datetime_columns(chunk)
                    chunks.append(chunk)
                    print(f"Loaded chunk with {len(chunk)} rows")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise e2
        
        if not chunks:
            raise ValueError("No data could be loaded from the CSV file")
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        print(f"Successfully loaded {len(df)} total rows")
        
        return df
    
    def _process_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process datetime columns after loading."""
        
        # Convert date columns properly
        date_columns = ['published', 'updated']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # Convert to datetime, handling various formats
                    df[col + '_date'] = pd.to_datetime(df[col], errors='coerce', utc=True)
                except Exception as e:
                    print(f"Warning: Could not parse {col} dates: {e}")
                    df[col + '_date'] = pd.NaT
        
        return df
    
    def _memory_safe_load(self, file_path: str) -> pd.DataFrame:
        """Memory-safe loading with proper error handling."""
        return self._load_csv(file_path)
    
    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load data with proper error handling."""
        try:
            if nrows:
                # For small samples, load directly
                df = pd.read_csv(
                    self.data_path,
                    nrows=nrows,
                    dtype=str,  # Load as string first
                    low_memory=False,
                    na_values=['', 'nan', 'NaN', 'null', 'NULL', 'None'],
                    keep_default_na=True,
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                df = self._process_datetime_columns(df)
            else:
                # For full dataset, use chunked loading
                df = self._memory_safe_load(self.data_path)
            
            self.df = df
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise


# Alternative quick fix for your existing code
def quick_fix_csv_loading():
    """
    Quick fix function you can use immediately.
    Replace your current loading code with this.
    """
    
    # Option 1: Load with all string types first
    try:
        df = pd.read_csv(
            "data/raw/articles.csv",
            dtype=str,  # Load everything as string
            low_memory=False,
            na_values=['', 'nan', 'NaN', 'null', 'NULL', 'None'],
            keep_default_na=True,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        # Convert date columns after loading
        date_columns = ['published', 'updated']
        for col in date_columns:
            if col in df.columns:
                df[col + '_date'] = pd.to_datetime(df[col], errors='coerce', utc=True)
        
        print(f"Successfully loaded {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Loading failed: {e}")
        return None


# Enhanced error handling for your current script
def load_with_error_handling(file_path: str, chunk_size: int = 100000):
    """
    Enhanced loading function with comprehensive error handling.
    """
    
    # Define expected column types for ArXiv data
    arxiv_dtypes = {
        'id': 'str',
        'title': 'str', 
        'summary': 'str',
        'author': 'str',
        'published': 'str',
        'updated': 'str',
        'primary_category': 'str',
        'category': 'str',
        'comment': 'str',
        'doi': 'str',
        'journal_ref': 'str'
    }
    
    chunks = []
    
    try:
        # Method 1: Try with explicit dtypes
        print("Attempting to load with explicit data types...")
        
        for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            dtype=arxiv_dtypes,
            low_memory=False,
            na_values=['', 'nan', 'NaN', 'null', 'NULL', 'None'],
            keep_default_na=True,
            encoding='utf-8',
            on_bad_lines='skip'
        ):
            # Process dates safely
            if 'published' in chunk.columns:
                chunk['published_date'] = pd.to_datetime(chunk['published'], errors='coerce', utc=True)
            if 'updated' in chunk.columns:
                chunk['updated_date'] = pd.to_datetime(chunk['updated'], errors='coerce', utc=True)
            
            chunks.append(chunk)
            print(f"Loaded chunk: {len(chunk)} rows")
            
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        try:
            # Method 2: Load everything as string first
            print("Attempting fallback: loading all as strings...")
            chunks = []
            
            for chunk in pd.read_csv(
                file_path,
                chunksize=chunk_size,
                dtype=str,
                low_memory=False,
                na_values=['', 'nan', 'NaN', 'null', 'NULL', 'None'],
                keep_default_na=True,
                encoding='utf-8',
                on_bad_lines='skip'
            ):
                # Process dates safely
                if 'published' in chunk.columns:
                    chunk['published_date'] = pd.to_datetime(chunk['published'], errors='coerce', utc=True)
                if 'updated' in chunk.columns:
                    chunk['updated_date'] = pd.to_datetime(chunk['updated'], errors='coerce', utc=True)
                
                chunks.append(chunk)
                print(f"Loaded chunk: {len(chunk)} rows")
                
        except Exception as e2:
            print(f"Method 2 also failed: {e2}")
            
            try:
                # Method 3: Most basic loading
                print("Attempting most basic loading...")
                chunks = []
                
                for chunk in pd.read_csv(
                    file_path,
                    chunksize=chunk_size,
                    dtype=str,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    engine='python'  # Use Python engine as fallback
                ):
                    chunks.append(chunk)
                    print(f"Loaded chunk: {len(chunk)} rows")
                    
            except Exception as e3:
                print(f"All methods failed: {e3}")
                raise e3
    
    if not chunks:
        raise ValueError("No data could be loaded")
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    return df


if __name__ == "__main__":
    # Test the fixes
    try:
        # Quick fix test
        print("Testing quick fix...")
        df = quick_fix_csv_loading()
        
        if df is not None:
            print(f"Success! Loaded {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Data types: {df.dtypes}")
            
            # Test with your cleaner
            from cleaner import EnhancedArxivDataCleaner
            
            cleaner = EnhancedArxivDataCleaner(use_spacy=False)  # Disable spacy for speed
            df_clean = cleaner.clean_dataset(df.head(1000), deep_clean=False)  # Test with small sample
            
            print(f"Cleaned data: {len(df_clean)} rows")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()