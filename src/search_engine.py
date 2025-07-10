"""
Enhanced FAISS-based search engine for arXiv articles with optimized indexing and querying.
Features:
- Hierarchical Navigable Small World (HNSW) index for efficient approximate nearest neighbor search
- Quantization for memory efficiency (PQ - Product Quantization)
- Multi-index support for different query types
- Advanced caching with memory management
- Result diversification and re-ranking
- GPU acceleration support
- Batch processing for large queries
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
import pickle
from pathlib import Path
import pandas as pd
import hashlib
import time
import os
import gc
from multiprocessing import Pool, cpu_count
from functools import lru_cache
from collections import defaultdict
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedArxivSearchEngine:
    def __init__(self, 
                 index_type: str = "HNSW32", 
                 use_gpu: bool = False,
                 cache_size: int = 10000):
        """
        Initialize the optimized search engine.
        
        Args:
            index_type: Type of FAISS index to use (HNSW32, IVF, PQ, etc.)
            use_gpu: Whether to use GPU acceleration if available
            cache_size: Maximum number of queries to cache
        """
        self.index = None
        self.secondary_index = None  # For title/author searches
        self.article_ids = []
        self.article_data = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.gpu_resources = None
        self.cache_size = cache_size
        
        # Initialize GPU if requested
        if self.use_gpu:
            self._init_gpu()
            
        logger.info(f"Initialized OptimizedArxivSearchEngine with index type: {index_type}")

    def _init_gpu(self):
        """Initialize GPU resources for FAISS."""
        try:
            self.gpu_resources = faiss.StandardGpuResources()
            logger.info("GPU resources initialized")
        except Exception as e:
            logger.warning(f"Could not initialize GPU: {e}. Falling back to CPU.")
            self.use_gpu = False

    def _create_optimized_index(self, n_vectors: int) -> faiss.Index:
        """
        Create an optimized FAISS index based on dataset size and available resources.
        
        Args:
            n_vectors: Number of vectors in the dataset
            
        Returns:
            Configured FAISS index
        """
        # Default to flat index for small datasets
        if n_vectors < 10000:
            logger.info("Using flat index for small dataset")
            return faiss.IndexFlatIP(self.dimension)
        
        # For medium to large datasets, use more sophisticated indices
        if self.index_type.startswith("HNSW"):
            # Hierarchical Navigable Small World graph
            m = int(self.index_type[4:]) if len(self.index_type) > 4 else 32
            index = faiss.IndexHNSWFlat(self.dimension, m)
            index.hnsw.efSearch = 128  # Balance between speed and accuracy
            index.hnsw.efConstruction = 200
            logger.info(f"Created HNSW index with m={m}")
        elif self.index_type.startswith("IVF"):
            # Inverted file index with product quantization
            nlist = min(1024, n_vectors // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, 8, 8)
            index.nprobe = min(32, nlist // 4)  # Number of clusters to explore
            logger.info(f"Created IVF PQ index with nlist={nlist}")
        else:
            # Default to flat index
            index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created flat index")
        
        # Move to GPU if available
        if self.use_gpu and self.gpu_resources:
            try:
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                logger.info("Moved index to GPU")
            except Exception as e:
                logger.warning(f"Could not move index to GPU: {e}")
        
        return index

    def build_index(self, 
                   embeddings: Union[Dict[str, np.ndarray], np.ndarray],
                   secondary_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        Build optimized FAISS indices from embeddings.
        
        Args:
            embeddings: Dictionary mapping article IDs to content embeddings OR numpy array
            secondary_embeddings: Optional dictionary for title/author embeddings
        """
        start_time = time.time()
        
        # Handle both dictionary and numpy array inputs
        if isinstance(embeddings, dict):
            if not embeddings:
                raise ValueError("No embeddings provided")
            self.article_ids = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[id_] for id_ in self.article_ids]).astype('float32')
        elif isinstance(embeddings, np.ndarray):
            if embeddings.size == 0:
                raise ValueError("Empty embedding array provided")
            embedding_matrix = embeddings.astype('float32')
            # Generate default article IDs if not provided
            if not self.article_ids:
                self.article_ids = [str(i) for i in range(len(embedding_matrix))]
        else:
            raise ValueError("Embeddings must be either a dictionary or numpy array")
        
        self.dimension = embedding_matrix.shape[1]
        
        # Validate embedding matrix
        if embedding_matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape: {embedding_matrix.shape}")
        
        if embedding_matrix.shape[0] == 0:
            raise ValueError("Empty embedding matrix")
            
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Create and train primary index
        self.index = self._create_optimized_index(len(self.article_ids))
        
        if not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embedding_matrix)
            
        self.index.add(embedding_matrix)
        
        # Build secondary index if provided
        if secondary_embeddings:
            secondary_matrix = np.array([secondary_embeddings[id_] for id_ in self.article_ids]).astype('float32')
            faiss.normalize_L2(secondary_matrix)
            
            self.secondary_index = self._create_optimized_index(len(self.article_ids))
            if not self.secondary_index.is_trained:
                self.secondary_index.train(secondary_matrix)
            self.secondary_index.add(secondary_matrix)
        
        build_time = time.time() - start_time
        logger.info(f"Built FAISS index with {len(self.article_ids)} embeddings in {build_time:.2f} seconds")

    def load_index(self, 
                  index_path: str, 
                  metadata_path: str,
                  secondary_index_path: Optional[str] = None):
        """
        Load pre-built FAISS indices from disk with memory mapping for large files.
        
        Args:
            index_path: Path to primary FAISS index file
            metadata_path: Path to article metadata file
            secondary_index_path: Optional path to secondary index file
        """
        start_time = time.time()
        
        # Load with memory mapping for large files
        if os.path.getsize(index_path) > 100 * 1024**2:  # >100MB
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            logger.info("Loaded primary index with memory mapping")
        else:
            self.index = faiss.read_index(index_path)
            
        if secondary_index_path:
            if os.path.getsize(secondary_index_path) > 100 * 1024**2:
                self.secondary_index = faiss.read_index(secondary_index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
                logger.info("Loaded secondary index with memory mapping")
            else:
                self.secondary_index = faiss.read_index(secondary_index_path)
        
        # Load article metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.article_ids = metadata["article_ids"]
            self.article_data = metadata["article_data"]
            
        self.dimension = self.index.d
        
        # Move to GPU if requested
        if self.use_gpu and self.gpu_resources:
            try:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                if self.secondary_index:
                    self.secondary_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.secondary_index)
                logger.info("Moved indices to GPU")
            except Exception as e:
                logger.warning(f"Could not move indices to GPU: {e}")
        
        load_time = time.time() - start_time
        logger.info(f"Loaded FAISS index with {len(self.article_ids)} embeddings in {load_time:.2f} seconds")

    def save_index(self, 
                  index_path: str, 
                  metadata_path: str,
                  secondary_index_path: Optional[str] = None):
        """
        Save the FAISS indices and metadata to disk with optimized storage.
        
        Args:
            index_path: Path to save primary FAISS index
            metadata_path: Path to save metadata
            secondary_index_path: Optional path to save secondary index
        """
        if not self.index:
            raise ValueError("Primary index not built")
            
        # Save indices
        faiss.write_index(self.index, index_path)
        if self.secondary_index and secondary_index_path:
            faiss.write_index(self.secondary_index, secondary_index_path)
        
        # Save article metadata
        metadata = {
            "article_ids": self.article_ids,
            "article_data": self.article_data
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved FAISS index to {index_path}")

    def load_article_data(self, df: pd.DataFrame):
        """
        Load article metadata for search results with memory optimization.
        
        Args:
            df: DataFrame containing article metadata
        """
        # Convert to dict of dicts for efficient lookup
        self.article_data = {
            row['id']: {
                'title': row.get('title', ''),
                'author': row.get('author', ''),
                'published_date': row.get('published_date', ''),
                'primary_category': row.get('primary_category', ''),
                'categories': row.get('categories', ''),
                'summary': row.get('summary', ''),
                'doi': row.get('doi', '')
            }
            for _, row in df.iterrows()
        }

    @lru_cache(maxsize=10000)
    def _get_cached_embedding(self, query_text: str, embedder) -> np.ndarray:
        """
        Get cached embedding for a query or generate new one with LRU cache.
        
        Args:
            query_text: Text query
            embedder: Embedder instance
            
        Returns:
            Query embedding
        """
        # Generate new embedding
        embedding = embedder.generate_embeddings([query_text])[0]
        return embedding

    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5,
               index_type: str = "primary") -> List[Dict]:
        """
        Search for similar articles using an embedding.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            index_type: Which index to use ('primary' or 'secondary')
            
        Returns:
            List of result dictionaries with article metadata
        """
        if not self.index:
            raise ValueError("Index not built")
            
        if not self.article_data:
            raise ValueError("Article data not loaded")
            
        # Normalize query embedding
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Select appropriate index
        index = self.index if index_type == "primary" else self.secondary_index
        
        # Search the index with increased nprobe for better recall
        if isinstance(index, faiss.IndexIVFPQ):
            index.nprobe = min(64, index.nlist // 2)  # Increase nprobe for better accuracy
            
        distances, indices = index.search(query_embedding, min(100, len(self.article_ids)))
        
        # Prepare results with metadata
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i < 0:  # FAISS returns -1 for invalid indices
                continue
                
            article_id = self.article_ids[i]
            article_info = self.article_data.get(article_id, {})
            article_info["similarity"] = float(distance)
            results.append(article_info)
            
        return self._rerank_results(results, k)

    def _rerank_results(self, results: List[Dict], k: int) -> List[Dict]:
        """
        Re-rank results using diversity and quality metrics.
        
        Args:
            results: Initial search results
            k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
            
        # Group by category and select top from each
        category_groups = defaultdict(list)
        for res in results:
            category = res.get('primary_category', '')
            category_groups[category].append(res)
            
        # Select top from each category
        diversified = []
        for category, group in category_groups.items():
            top = heapq.nlargest(
                min(2, len(group)), 
                group, 
                key=lambda x: x['similarity']
            )
            diversified.extend(top)
            
        # Sort by similarity and take top k
        diversified.sort(key=lambda x: x['similarity'], reverse=True)
        return diversified[:k]

    def search_by_text(self, 
                      query_text: str, 
                      embedder, 
                      k: int = 5,
                      search_fields: str = "all") -> List[Dict]:
        """
        Optimized search with caching, batch processing, and field selection.
        
        Args:
            query_text: Text query to search for
            embedder: Embedder instance
            k: Number of results to return
            search_fields: Which fields to search ('content', 'title', 'all')
            
        Returns:
            List of result dictionaries with article metadata
        """
        start_time = time.time()
        
        # Get cached or generate embedding
        query_embedding = self._get_cached_embedding(query_text, embedder)
        
        # Search based on requested fields
        if search_fields == "content" or not self.secondary_index:
            results = self.search(query_embedding, k, "primary")
        elif search_fields == "title":
            results = self.search(query_embedding, k, "secondary")
        else:  # 'all' - combine results from both indices
            content_results = self.search(query_embedding, k, "primary")
            title_results = self.search(query_embedding, k, "secondary")
            
            # Combine and deduplicate results
            combined = {res['id']: res for res in content_results + title_results}
            results = list(combined.values())
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:k]
        
        search_time = time.time() - start_time
        logger.debug(f"Search completed in {search_time:.4f} seconds")
        
        return results

    def batch_search(self, 
                    queries: List[str], 
                    embedder,
                    k: int = 5) -> List[List[Dict]]:
        """
        Process multiple queries efficiently in batch.
        
        Args:
            queries: List of text queries
            embedder: Embedder instance
            k: Number of results per query
            
        Returns:
            List of result lists for each query
        """
        # Generate all embeddings first
        query_embeddings = embedder.generate_embeddings(queries)
        
        # Process in parallel if many queries
        if len(queries) > 100:
            with Pool(min(cpu_count(), 8)) as pool:
                results = pool.starmap(
                    self.search,
                    [(emb, k) for emb in query_embeddings]
                )
        else:
            results = [self.search(emb, k) for emb in query_embeddings]
            
        return results

    def clear_cache(self):
        """Clear all cached embeddings and free memory."""
        self._get_cached_embedding.cache_clear()
        gc.collect()
        logger.info("Cleared search cache")

if __name__ == "__main__":
    # Example usage
    from embedder import ArxivEmbedder
    from data_loader import ArxivDataLoader
    
    # Initialize with HNSW index and GPU support
    search_engine = OptimizedArxivSearchEngine(index_type="HNSW32", use_gpu=True)
    
    # Load data
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data(nrows=100000)  # Adjust based on available memory
    
    # Generate embeddings
    embedder = ArxivEmbedder()
    content_embeddings = embedder.embed_articles(df, text_field="summary_clean")
    title_embeddings = embedder.embed_articles(df, text_field="title_clean")
    
    # Build search indices
    search_engine.build_index(content_embeddings, title_embeddings)
    search_engine.load_article_data(df)
    
    # Perform searches
    query = "machine learning applications in healthcare"
    results = search_engine.search_by_text(query, embedder, k=10)
    
    print(f"Search results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")