"""
Enhanced module for generating semantic embeddings of arXiv articles using sentence transformers.
Optimized for large datasets (3.5GB+) with memory-efficient processing and advanced caching.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
import pickle
import gc
import psutil
import faiss
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedArxivEmbedder:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "data/embeddings",
                 max_cache_size: int = 4):
        """
        Enhanced embedder with memory optimization and advanced caching.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to store embeddings and indexes
            max_cache_size: Maximum cache size in GB
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size * (1024**3)  # Convert to bytes
        self.current_cache_size = 0
        self.embedding_cache = {}
        self.model = self._load_model()
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Memory monitoring
        self.memory_monitor = False
        self.max_memory_usage = 0.8  # Max memory usage before cleanup
        
        logger.info(f"Initialized EnhancedArxivEmbedder with model: {model_name}")

    def _load_model(self) -> SentenceTransformer:
        """Load model with memory optimization."""
        try:
            # Try to load with GPU first, fallback to CPU
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading model on {device.upper()}")
            model = SentenceTransformer(self.model_name, device=device)
            
            # Set eval mode and disable gradient calculation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
            return model
        except Exception as e:
            logger.warning(f"Couldn't load model on GPU, falling back to CPU: {e}")
            return SentenceTransformer(self.model_name, device='cpu')

    def _check_memory(self) -> bool:
        """Check if memory usage is too high."""
        mem = psutil.virtual_memory()
        return mem.used / mem.total > self.max_memory_usage

    def _cleanup_cache(self):
        """Clean up cache when memory is low."""
        logger.warning("Memory pressure high - cleaning embedding cache")
        self.embedding_cache.clear()
        gc.collect()
        self.current_cache_size = 0

    def _get_cache_key(self, text: str) -> str:
        """Generate consistent cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def generate_embeddings(self, 
                          texts: List[str], 
                          batch_size: int = 128,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings with memory-efficient batch processing.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        
        # Check for cached results first
        cached_results = []
        uncached_texts = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_results.append(self.embedding_cache[cache_key])
            else:
                uncached_texts.append(text)
                cache_indices.append(i)
        
        # Process uncached texts in batches
        uncached_embeddings = []
        total_batches = (len(uncached_texts)) // batch_size + 1
        
        for batch_idx in range(0, len(uncached_texts), batch_size):
            if self._check_memory():
                self._cleanup_cache()
                
            batch_texts = uncached_texts[batch_idx:batch_idx + batch_size]
            
            # Generate embeddings
            start_time = time.time()
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            processing_time = time.time() - start_time
            
            if show_progress:
                logger.info(f"Processed batch {batch_idx//batch_size + 1}/{total_batches} "
                          f"({len(batch_texts)} texts) in {processing_time:.2f}s")
            
            # Cache results
            for text, embedding in zip(batch_texts, batch_embeddings):
                cache_key = self._get_cache_key(text)
                self.embedding_cache[cache_key] = embedding
                self.current_cache_size += embedding.nbytes
                
                # Check cache size
                if self.current_cache_size > self.max_cache_size:
                    self._cleanup_cache()
            
            uncached_embeddings.append(batch_embeddings)
        
        # Combine cached and uncached results
        if uncached_embeddings:
            all_uncached = np.concatenate(uncached_embeddings)
        else:
            all_uncached = np.array([])
            
        if cached_results:
            all_cached = np.stack(cached_results)
        else:
            all_cached = np.array([])
        
        # Reconstruct original order
        final_embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        if all_cached.size > 0:
            cached_indices = [i for i in range(len(texts)) if i not in cache_indices]
            final_embeddings[cached_indices] = all_cached
        if all_uncached.size > 0:
            final_embeddings[cache_indices] = all_uncached
        
        return final_embeddings

    def embed_articles(self, 
                      df: pd.DataFrame, 
                      text_field: str = "summary_clean",
                      batch_size: int = 1024,
                      parallel: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all articles with optimized processing.
        
        Args:
            df: DataFrame containing articles
            text_field: Column name containing text to embed
            batch_size: Number of articles per batch
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary mapping article IDs to embeddings
        """
        if text_field not in df.columns:
            raise ValueError(f"Text field '{text_field}' not found in DataFrame")
            
        logger.info(f"Generating embeddings for {len(df)} articles")
        
        # Prepare data
        texts = df[text_field].fillna("").astype(str).tolist()
        ids = df["id"].astype(str).tolist()
        
        if parallel and len(texts) > 10000:
            return self._parallel_embed(texts, ids, batch_size)
        
        # Single process embedding
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        return {id_: embedding for id_, embedding in zip(ids, embeddings)}

    def _parallel_embed(self, 
                       texts: List[str], 
                       ids: List[str],
                       batch_size: int) -> Dict[str, np.ndarray]:
        """
        Parallel embedding generation for large datasets.
        """
        logger.info("Using parallel processing for large dataset")
        
        # Split data into chunks for parallel processing
        num_workers = min(cpu_count(), 8)  # Don't use all cores to avoid memory issues
        chunk_size = len(texts) // num_workers
        chunks = [
            (texts[i:i + chunk_size], ids[i:i + chunk_size])
            for i in range(0, len(texts), chunk_size)
        ]
        
        # Process chunks in parallel
        with Pool(num_workers) as pool:
            results = pool.starmap(
                self._embed_chunk,
                [(chunk_texts, chunk_ids, batch_size) 
                 for chunk_texts, chunk_ids in chunks]
            )
        
        # Combine results
        embeddings = {}
        for result in results:
            embeddings.update(result)
            
        return embeddings

    def _embed_chunk(self, 
                    texts: List[str], 
                    ids: List[str],
                    batch_size: int) -> Dict[str, np.ndarray]:
        """
        Worker function for parallel embedding.
        """
        embeddings = self.generate_embeddings(texts, batch_size=batch_size, show_progress=False)
        return {id_: embedding for id_, embedding in zip(ids, embeddings)}

    def save_embeddings(self, 
                       embeddings: Dict[str, np.ndarray], 
                       output_dir: str,
                       save_index: bool = True):
        """
        Save embeddings to disk with optimized storage format.
        
        Args:
            embeddings: Dictionary of embeddings
            output_dir: Directory to save embeddings
            save_index: Whether to also save FAISS index
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings in efficient format
        emb_path = output_path / f"embeddings_{self.model_name}.pkl"
        with open(emb_path, "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        logger.info(f"Saved embeddings to {emb_path}")
        
        # Save FAISS index if requested
        if save_index:
            self.save_faiss_index(embeddings, output_path)

    def save_faiss_index(self, 
                        embeddings: Dict[str, np.ndarray], 
                        output_dir: Path):
        """
        Build and save optimized FAISS index for fast similarity search.
        
        Args:
            embeddings: Dictionary of embeddings
            output_dir: Directory to save index
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        logger.info("Building optimized FAISS index")
        
        # Convert embeddings to numpy array
        ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[id_] for id_ in ids]).astype('float32')
        
        # Create optimized index
        index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Add vectors to index
        index.add(embedding_matrix)
        
        # Save index
        index_path = output_dir / f"faiss_index_{self.model_name}.index"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = output_dir / f"faiss_metadata_{self.model_name}.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({"article_ids": ids}, f)
            
        logger.info(f"Saved FAISS index to {index_path}")

    def load_embeddings(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from disk with memory mapping for large files.
        
        Args:
            input_path: Path to embeddings file
            
        Returns:
            Dictionary of embeddings
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")
            
        # Check file size
        file_size = input_path.stat().st_size / (1024**3)  # GB
        
        if file_size > 2:  # For very large files, use memory mapping
            logger.info(f"Loading large embeddings file ({file_size:.2f} GB) with memory mapping")
            with open(input_path, "rb") as f:
                embeddings = pickle.load(f, buffers=None)  # Disable buffer for mmap
        else:
            with open(input_path, "rb") as f:
                embeddings = pickle.load(f)
                
        logger.info(f"Loaded embeddings from {input_path}")
        return embeddings

    def load_faiss_index(self, 
                        index_path: str, 
                        metadata_path: str) -> Tuple[faiss.Index, List[str]]:
        """
        Load FAISS index and metadata.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
            
        Returns:
            Tuple of (FAISS index, article IDs)
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        # Load index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            article_ids = metadata["article_ids"]
            
        logger.info(f"Loaded FAISS index with {len(article_ids)} embeddings")
        return index, article_ids

    def clear_cache(self):
        """Clear all cached embeddings and free memory."""
        self.embedding_cache.clear()
        gc.collect()
        self.current_cache_size = 0
        logger.info("Cleared embedding cache")

if __name__ == "__main__":
    # Example usage
    from data_loader import ArxivDataLoader
    
    # Initialize embedder with large cache
    embedder = EnhancedArxivEmbedder(
        model_name="all-MiniLM-L6-v2",
        cache_dir="data/embeddings",
        max_cache_size=8  # GB
    )
    
    # Load sample data
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data(nrows=10000)  # Adjust based on available memory
    
    # Generate embeddings
    embeddings = embedder.embed_articles(df, parallel=True)
    
    # Save embeddings and index
    embedder.save_embeddings(embeddings, "data/embeddings/")
    
    # Memory cleanup
    embedder.clear_cache()