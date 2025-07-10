# """
# Enhanced data loading and management module for ArXiv chatbot project.
# Optimized for large datasets (3.5GB+) with memory-efficient processing.
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime
# import logging
# from typing import Optional, Dict, List, Tuple, Union
# import os
# from pathlib import Path
# import pickle
# import dask.dataframe as dd
# import pyarrow.parquet as pq
# import pyarrow as pa
# from multiprocessing import Pool, cpu_count
# import gc
# import psutil
# import zstandard as zstd
# import tempfile
# import shutil
# import time
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global cache for parsed data with size limit
# _DATA_CACHE = {}
# _MAX_CACHE_SIZE = 2  # GB
# _CACHE_SIZE = 0

# class ArxivDataLoader:
#     """
#     Enhanced data loader optimized for large arXiv datasets.
#     Features:
#     - Memory-efficient processing with chunking
#     - Multiple file format support (CSV, Parquet, Zstd)
#     - Parallel processing
#     - Smart caching with size limits
#     - Automatic memory management
#     """
    
#     def __init__(self, data_path: str = "data/raw/articles.parquet"):
#         """
#         Initialize the data loader with enhanced capabilities.
        
#         Args:
#             data_path: Path to the data file (supports .csv, .parquet, .zst)
#         """
#         self.data_path = Path(data_path)
#         self.df = None
#         self.metadata = {}
#         self._current_chunk = None
#         self._chunk_index = 0
#         self._dask_df = None
        
#         # Memory management
#         self.max_memory = psutil.virtual_memory().available * 0.7  # Use 70% of available RAM
#         self.chunksize = self._calculate_optimal_chunksize()
        

#     def _calculate_optimal_chunksize(self) -> int:
#         """Calculate optimal chunk size based on available memory."""
#         # Estimate memory usage per row (conservative estimate)
#         bytes_per_row = 2000  # Approximate bytes per row in arXiv data
#         available_mem = self.max_memory
        
#         # Target chunk size (aim for chunks that use 20% of available memory)
#         target_chunk_mem = available_mem * 0.2
#         chunksize = int(target_chunk_mem // bytes_per_row)
        
#         # Set reasonable bounds
#         chunksize = max(10000, min(chunksize, 500000))
        
#         logger.info(f"Calculated optimal chunksize: {chunksize:,}")
#         return chunksize
    
#     def _memory_safe_load(self, file_path: Path) -> pd.DataFrame:
#         """
#         Memory-safe loading of data with automatic format detection.
#         """
#         if not file_path.exists():
#             raise FileNotFoundError(f"Data file not found: {file_path}")
            
#         # Check file extension for format
#         ext = file_path.suffix.lower()
        
#         if ext == '.parquet':
#             return self._load_parquet(file_path)
#         elif ext == '.zst':
#             return self._load_zstd(file_path)
#         else:  # Default to CSV
#             return self._load_csv(file_path)
    
#     def _load_csv(self, file_path: Path) -> pd.DataFrame:
#         """Optimized CSV loading with proper dtypes and chunking."""
#         # First, peek at the file to understand structure
#         logger.info("Determining optimal data types...")
        
#         # Read a small sample to understand the data structure
#         try:
#             sample = pd.read_csv(file_path, nrows=1000, dtype=str)  # Read everything as string first
#         except Exception as e:
#             logger.error(f"Error reading CSV sample: {e}")
#             raise
        
#         # Define explicit dtypes, ensuring date columns are strings
#         dtypes = self._infer_safe_dtypes(sample)
        
#         # Read in chunks with safe dtypes
#         logger.info(f"Loading CSV in chunks of {self.chunksize:,} rows...")
#         chunks = []
        
#         try:
#             for chunk in pd.read_csv(
#                 file_path,
#                 dtype=dtypes,
#                 chunksize=self.chunksize,
#                 parse_dates=False,  # Never auto-parse dates
#                 on_bad_lines='skip',
#                 low_memory=False,
#                 na_values=['', 'NA', 'NULL', 'null', 'None'],
#                 keep_default_na=True
#             ):
#                 # Manually convert date columns after loading
#                 chunk = self._convert_date_columns(chunk)
#                 chunks.append(chunk)
                
#                 if self._memory_pressure_high():
#                     self._cleanup_memory(chunks)
                    
#         except Exception as e:
#             logger.error(f"Error during CSV loading: {e}")
#             raise
            
#         if not chunks:
#             raise ValueError("No data could be loaded from CSV file")
            
#         df = pd.concat(chunks, ignore_index=True)
#         return df
    
#     def _infer_safe_dtypes(self, sample: pd.DataFrame) -> Dict[str, str]:
#         """Infer safe dtypes, keeping date columns as strings."""
#         dtypes = {}
        
#         # Define known date columns that should be kept as strings
#         date_columns = {'published', 'updated', 'published_date', 'updated_date'}
        
#         for col in sample.columns:
#             col_lower = col.lower()
            
#             # Force date columns to string
#             if col in date_columns or col_lower in date_columns:
#                 dtypes[col] = 'string'
#                 continue
                
#             # Check for date-like patterns in column names
#             if any(date_word in col_lower for date_word in ['date', 'time', 'publish', 'update']):
#                 dtypes[col] = 'string'
#                 continue
            
#             # Check if values look like dates
#             if self._looks_like_date_column(sample[col]):
#                 dtypes[col] = 'string'
#                 continue
                
#             # For other columns, infer safe types
#             if sample[col].dtype == 'object':
#                 # Check if it's purely numeric
#                 try:
#                     pd.to_numeric(sample[col], errors='raise')
#                     dtypes[col] = 'float64'
#                 except:
#                     dtypes[col] = 'string'
#             else:
#                 dtypes[col] = str(sample[col].dtype)

#         return dtypes
    
#     def _looks_like_date_column(self, series: pd.Series) -> bool:
#         """Check if a series contains date-like strings."""
#         if series.dtype != 'object':
#             return False
            
#         # Sample a few non-null values
#         sample_values = series.dropna().head(10)
#         if len(sample_values) == 0:
#             return False
            
#         date_patterns = [
#             r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
#             r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
#             r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
#         ]
        
#         for value in sample_values:
#             if isinstance(value, str):
#                 for pattern in date_patterns:
#                     if re.search(pattern, value):
#                         return True
#         return False
    
#     def _convert_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Safely convert date columns after loading."""
#         date_columns = ['published', 'updated']
        
#         for col in date_columns:
#             if col in df.columns:
#                 try:
#                     df[f'{col}_date'] = pd.to_datetime(df[col], errors='coerce')
#                 except Exception as e:
#                     logger.warning(f"Could not convert {col} to datetime: {e}")
#                     df[f'{col}_date'] = pd.NaT
        
#         return df
    
#     def _load_parquet(self, file_path: Path) -> pd.DataFrame:
#         """Load parquet file with memory optimization."""
#         logger.info("Loading Parquet file...")
        
#         # Use pyarrow for memory mapping
#         table = pq.read_table(file_path)
#         df = table.to_pandas()
        
#         # Convert date columns
#         df = self._convert_date_columns(df)
            
#         return df
    
#     def _load_zstd(self, file_path: Path) -> pd.DataFrame:
#         """Load Zstandard compressed file."""
#         logger.info("Loading Zstd compressed file...")
        
#         # Create temp file for decompression
#         with tempfile.NamedTemporaryFile(delete=False) as tmp:
#             try:
#                 # Decompress
#                 with open(file_path, 'rb') as fh:
#                     dctx = zstd.ZstdDecompressor()
#                     with dctx.stream_reader(fh) as reader:
#                         shutil.copyfileobj(reader, tmp)
                
#                 # Now load the decompressed file
#                 tmp_path = Path(tmp.name)
#                 if tmp_path.suffix == '.parquet':
#                     return self._load_parquet(tmp_path)
#                 else:
#                     return self._load_csv(tmp_path)
#             finally:
#                 tmp_path.unlink(missing_ok=True)
    
#     def _memory_pressure_high(self) -> bool:
#         """Check if memory usage is too high."""
#         mem = psutil.virtual_memory()
#         return mem.used / mem.total > 0.8
    
#     def _cleanup_memory(self, chunks: List[pd.DataFrame]):
#         """Clean up memory during loading."""
#         logger.warning("Memory pressure high - performing cleanup...")
#         for chunk in chunks:
#             del chunk
#         gc.collect()
    
#     def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
#         """
#         Enhanced data loading with memory optimization and caching.
        
#         Args:
#             nrows: Number of rows to load (None for all)
            
#         Returns:
#             DataFrame with loaded data
#         """
#         cache_key = f"{self.data_path}_{nrows}"
        
#         # Check cache with size consideration
#         if cache_key in _DATA_CACHE:
#             cached_size = _DATA_CACHE[cache_key].memory_usage(deep=True).sum() / (1024**3)
#             if cached_size < _MAX_CACHE_SIZE:
#                 logger.info(f"Using cached data ({cached_size:.2f} GB)")
#                 self.df = _DATA_CACHE[cache_key]
#                 self._compute_metadata()
#                 return self.df
#             else:
#                 del _DATA_CACHE[cache_key]
#                 gc.collect()
                
#         try:
#             logger.info(f"Loading data from {self.data_path}")
#             start_time = time.time()
            
#             # Use safe loading approach for all formats
#             self.df = self._memory_safe_load(self.data_path)
            
#             if nrows:
#                 self.df = self.df.head(nrows)
            
#             # Cache the processed data if it fits
#             df_size = self.df.memory_usage(deep=True).sum() / (1024**3)
#             if df_size < _MAX_CACHE_SIZE:
#                 _DATA_CACHE[cache_key] = self.df.copy()
#                 global _CACHE_SIZE
#                 _CACHE_SIZE += df_size
#                 logger.info(f"Cached data ({df_size:.2f} GB), total cache: {_CACHE_SIZE:.2f} GB")
            
#             load_time = time.time() - start_time
#             logger.info(f"Loaded {len(self.df)} articles in {load_time:.2f} seconds")
            
#             self._compute_metadata()
#             return self.df
            
#         except Exception as e:
#             logger.error(f"Error loading data: {e}")
#             raise

#     def load_iter(self, batch_size: int = 10000) -> "Iterator[pd.DataFrame]":
#         """
#         Iteratively load data in batches for memory efficiency.
        
#         Args:
#             batch_size: Number of rows per batch
            
#         Returns:
#             Iterator of DataFrames
#         """
#         if not self.data_path.exists():
#             raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
#         if self.data_path.suffix == '.parquet':
#             # Use pyarrow's batch reading
#             table = pq.read_table(self.data_path)
#             for batch in table.to_batches(max_chunksize=batch_size):
#                 yield batch.to_pandas()
#         else:
#             # Use pandas chunking for CSV with safe dtypes
#             sample = pd.read_csv(self.data_path, nrows=1000, dtype=str)
#             dtypes = self._infer_safe_dtypes(sample)
            
#             for chunk in pd.read_csv(
#                 self.data_path,
#                 chunksize=batch_size,
#                 dtype=dtypes,
#                 parse_dates=False
#             ):
#                 yield self._convert_date_columns(chunk)
    
#     def _compute_metadata(self):
#         """Compute enhanced metadata about the dataset."""
#         if self.df is None:
#             return
        
#         # Compute metadata efficiently using sampling for large datasets
#         sample_size = min(10000, len(self.df))
#         sample = self.df.sample(n=sample_size) if len(self.df) > sample_size else self.df
        
#         # Use safe column access
#         date_col = 'published_date' if 'published_date' in self.df.columns else None
#         category_col = 'primary_category' if 'primary_category' in self.df.columns else 'category'
        
#         self.metadata = {
#             'total_articles': len(self.df),
#             'columns': list(self.df.columns),
#             'missing_values': {
#                 col: self.df[col].isna().sum() 
#                 for col in self.df.columns
#             },
#             'memory_usage': self.df.memory_usage(deep=True).sum() / (1024**2),  # MB
#         }
        
#         # Add date range if date column exists
#         if date_col and date_col in self.df.columns:
#             self.metadata['date_range'] = {
#                 'start': self.df[date_col].min(),
#                 'end': self.df[date_col].max()
#             }
        
#         # Add category info if available
#         if category_col in self.df.columns:
#             self.metadata['unique_categories'] = self.df[category_col].nunique()
#             self.metadata['top_categories'] = self.df[category_col].value_counts().head(10).to_dict()
        
#         # Add sample statistics
#         if 'title' in self.df.columns:
#             self.metadata['avg_title_length'] = sample['title'].str.len().mean()
#         if 'summary' in self.df.columns:
#             self.metadata['avg_summary_length'] = sample['summary'].str.len().mean()
#         if 'author' in self.df.columns:
#             self.metadata['unique_authors'] = self.df['author'].nunique()
    
#     def get_metadata(self) -> Dict:
#         """Get enhanced dataset metadata."""
#         return self.metadata
    
#     def filter_by_category(self, categories: List[str]) -> pd.DataFrame:
#         """
#         Enhanced filter by category with memory optimization.
#         """
#         if self.df is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         # Find the correct category column
#         category_col = None
#         for col in ['primary_category', 'category', 'categories']:
#             if col in self.df.columns:
#                 category_col = col
#                 break
                
#         if category_col is None:
#             raise ValueError("No category column found in data")
        
#         # Use categorical dtype if possible
#         if not pd.api.types.is_categorical_dtype(self.df[category_col]):
#             self.df[category_col] = self.df[category_col].astype('category')
        
#         mask = self.df[category_col].isin(categories)
#         return self.df[mask].copy()
    
#     def filter_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
#         """
#         Enhanced date filtering with optimized date parsing.
#         """
#         if self.df is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         # Find the correct date column
#         date_col = None
#         for col in ['published_date', 'updated_date', 'date']:
#             if col in self.df.columns:
#                 date_col = col
#                 break
                
#         if date_col is None:
#             raise ValueError("No date column found in data")
        
#         start = pd.to_datetime(start_date, errors='coerce')
#         end = pd.to_datetime(end_date, errors='coerce')
        
#         if pd.isna(start) or pd.isna(end):
#             raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        
#         mask = (self.df[date_col] >= start) & (self.df[date_col] <= end)
#         return self.df[mask].copy()
    
#     def get_sample(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
#         """
#         Get a stratified random sample of articles.
#         """
#         if self.df is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         # Find category column
#         category_col = None
#         for col in ['primary_category', 'category', 'categories']:
#             if col in self.df.columns:
#                 category_col = col
#                 break
                
#         if category_col is None:
#             # If no category column, just return random sample
#             return self.df.sample(n=min(n, len(self.df)), random_state=random_state)
        
#         return self.df.groupby(category_col, group_keys=False).apply(
#             lambda x: x.sample(
#                 min(len(x), max(1, n // self.df[category_col].nunique())),
#                 random_state=random_state
#             )
#         )

#     def get_articles_by_author(self, author_pattern: str) -> pd.DataFrame:
#         """
#         Enhanced author search with optimized string operations.
#         """
#         if self.df is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         if 'author' not in self.df.columns:
#             raise ValueError("No author column found in data")
        
#         # Use vectorized string operations
#         mask = self.df['author'].str.contains(author_pattern, case=False, na=False, regex=True)
#         return self.df[mask].copy()
    
#     def get_top_categories(self, n: int = 10) -> pd.Series:
#         """
#         Get top N categories with optimized counting.
#         """
#         if self.df is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         # Find category column
#         category_col = None
#         for col in ['primary_category', 'category', 'categories']:
#             if col in self.df.columns:
#                 category_col = col
#                 break
                
#         if category_col is None:
#             raise ValueError("No category column found in data")
        
#         return self.df[category_col].value_counts().head(n)
    
#     def search_by_keywords(self, keywords: List[str], field: str = 'summary') -> pd.DataFrame:
#         """
#         Enhanced keyword search with parallel processing.
#         """
#         if self.df is None:
#             raise ValueError("Data not loaded. Call load_data() first.")
        
#         if field not in self.df.columns:
#             raise ValueError(f"Field '{field}' not found in data. Available fields: {list(self.df.columns)}")
        
#         # Prepare keywords for regex
#         keywords_regex = '|'.join(map(re.escape, keywords))
        
#         # Use parallel processing for large datasets
#         if len(self.df) > 100000:
#             with Pool(cpu_count()) as pool:
#                 masks = pool.starmap(
#                     self._search_worker,
#                     [(chunk, field, keywords_regex) for chunk in np.array_split(self.df, cpu_count() * 2)]
#                 )
#             mask = pd.concat(masks)
#         else:
#             mask = self.df[field].str.contains(keywords_regex, case=False, na=False)
        
#         return self.df[mask].copy()
    
#     def _search_worker(self, df_chunk: pd.DataFrame, field: str, pattern: str) -> pd.Series:
#         """Worker function for parallel search."""
#         return df_chunk[field].str.contains(pattern, case=False, na=False)
    
#     def export_data(self, df: pd.DataFrame, output_path: str, format: str = 'parquet'):
#         """
#         Enhanced data export with multiple format support.
#         """
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
        
#         if format == 'parquet':
#             df.to_parquet(output_path, engine='pyarrow')
#         elif format == 'csv':
#             df.to_csv(output_path, index=False)
#         elif format == 'zstd':
#             with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
#                 df.to_csv(tmp.name, index=False)
#                 with open(output_path, 'wb') as fh:
#                     cctx = zstd.ZstdCompressor()
#                     with cctx.stream_writer(fh) as writer:
#                         with open(tmp.name, 'rb') as f_in:
#                             shutil.copyfileobj(f_in, writer)
#         else:
#             raise ValueError("Format must be 'parquet', 'csv', or 'zstd'")
        
#         logger.info(f"Exported {len(df)} articles to {output_path} ({format})")


# class DataSplitter:
#     """
#     Enhanced data splitting utility with memory optimization.
#     """
    
#     @staticmethod
#     def split_by_date(df: pd.DataFrame, 
#                      train_ratio: float = 0.7, 
#                      val_ratio: float = 0.15,
#                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """
#         Memory-efficient date-based splitting.
#         """
#         if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
#             raise ValueError("Ratios must be between 0 and 1")
#         if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
#             raise ValueError("Ratios must sum to 1.0")
        
#         # Find date column
#         date_col = None
#         for col in ['published_date', 'updated_date', 'date']:
#             if col in df.columns:
#                 date_col = col
#                 break
                
#         if date_col is None:
#             raise ValueError("No date column found for splitting")
        
#         # Sort by date (memory efficient)
#         df_sorted = df.sort_values(date_col)
        
#         # Calculate split points
#         n = len(df_sorted)
#         train_end = int(n * train_ratio)
#         val_end = int(n * (train_ratio + val_ratio))
        
#         # Split without copying when possible
#         train_df = df_sorted.iloc[:train_end].copy()
#         val_df = df_sorted.iloc[train_end:val_end].copy()
#         test_df = df_sorted.iloc[val_end:].copy()
        
#         return train_df, val_df, test_df
    
#     @staticmethod
#     def split_random(df: pd.DataFrame, 
#                     train_ratio: float = 0.7, 
#                     val_ratio: float = 0.15,
#                     test_ratio: float = 0.15,
#                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """
#         Memory-efficient random splitting with chunking.
#         """
#         if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
#             raise ValueError("Ratios must be between 0 and 1")
#         if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
#             raise ValueError("Ratios must sum to 1.0")
        
#         # Calculate split points
#         n = len(df)
#         train_end = int(n * train_ratio)
#         val_end = int(n * (train_ratio + val_ratio))
        
#         # Shuffle indices instead of whole DataFrame
#         indices = np.arange(n)
#         np.random.seed(random_state)
#         np.random.shuffle(indices)
        
#         # Split using indices
#         train_df = df.iloc[indices[:train_end]].copy()
#         val_df = df.iloc[indices[train_end:val_end]].copy()
#         test_df = df.iloc[indices[val_end:]].copy()
        
#         return train_df, val_df, test_df


# if __name__ == "__main__":
#     # Example usage
#     loader = ArxivDataLoader("data/raw/articles.csv")  # Changed to CSV for testing
    
#     # Load sample data with memory optimization
#     df = loader.load_data(nrows=1000)
    
#     # Print enhanced metadata
#     print("Dataset metadata:")
#     print(loader.get_metadata())
    
#     # Get top categories if available
#     try:
#         print("\nTop categories:")
#         print(loader.get_top_categories())
#     except ValueError as e:
#         print(f"Categories not available: {e}")
    
#     # Search example with parallel processing
#     try:
#         ml_articles = loader.search_by_keywords(['machine learning', 'neural network'], 'summary')
#         print(f"\nFound {len(ml_articles)} articles about machine learning")
#     except ValueError as e:
#         print(f"Search not available: {e}")

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