"""
Enhanced module for generating semantic embeddings of arXiv articles with optimized chatbot support.
Combines large-scale processing capabilities with real-time query functionality.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import pickle
import gc
import psutil
import faiss
import hashlib
from multiprocessing import Pool, cpu_count
import time
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Dataclass for standardized search results"""
    article_id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict] = None

class ArxivEmbedder:
    """
    Dual-mode embedder supporting both:
    1. Large-scale batch processing of arXiv datasets
    2. Real-time query embedding and search for chatbot
    
    Key Features:
    - Memory-efficient batch processing with smart caching
    - FAISS-based similarity search
    - Real-time query processing
    - Context retrieval for chatbot responses
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "data/embeddings",
                 max_cache_size: int = 4):
        """
        Initialize embedder with optimized settings for both batch and real-time use.
        
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
        
        # FAISS index and metadata
        self.index = None
        self.article_ids = []
        self.article_metadata = None
        
        # Load model with hardware optimization
        self.model = self._load_model()
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Memory management
        self.memory_monitor = True
        self.max_memory_usage = 0.8
        
        logger.info(f"Initialized ArxivEmbedder with model: {model_name}")

    def _load_model(self) -> SentenceTransformer:
        """Load model with GPU/CPU optimization and eval mode"""
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading model on {device.upper()}")
            model = SentenceTransformer(self.model_name, device=device)
            
            # Optimize model for inference
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
            return model
        except Exception as e:
            logger.warning(f"Couldn't load model on GPU, falling back to CPU: {e}")
            return SentenceTransformer(self.model_name, device='cpu')

    # --------------------------
    # Core Embedding Functionality
    # --------------------------
    
    def embed_texts(self, 
                   texts: List[str], 
                   batch_size: int = 128,
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts with memory-efficient processing.
        
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
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        return {id_: embedding for id_, embedding in zip(ids, embeddings)}

    # --------------------------
    # Chatbot-Specific Methods
    # --------------------------
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query for real-time search.
        
        Args:
            query: User's natural language query
            
        Returns:
            Embedding vector for the query
        """
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        # Generate embedding
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Cache result
        self.embedding_cache[cache_key] = embedding
        self.current_cache_size += embedding.nbytes
        
        return embedding

    def search_similar(self, 
                      query: str, 
                      top_k: int = 5,
                      return_content: bool = False) -> List[SearchResult]:
        """
        Find most similar articles to user query using FAISS index.
        
        Args:
            query: User's search query
            top_k: Number of results to return
            return_content: Whether to include article content
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("FAISS index not loaded. Call load_faiss_index() first.")
            
        # Embed the query
        query_embedding = self.embed_query(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
                
            article_id = self.article_ids[idx]
            result = SearchResult(
                article_id=article_id,
                score=float(score),
                metadata=self.article_metadata.get(article_id, {}) if self.article_metadata else None
            )
            
            if return_content:
                result.content = self._get_article_content(article_id)
                
            results.append(result)
            
        return results

    def get_context(self, 
                   query: str,
                   top_k: int = 3,
                   max_length: int = 2000) -> str:
        """
        Get relevant context for chatbot response generation.
        
        Args:
            query: User's query
            top_k: Number of articles to include
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        results = self.search_similar(query, top_k=top_k, return_content=True)
        
        context_parts = []
        for result in results:
            if result.content:
                context_parts.append(
                    f"Article {result.article_id} (relevance: {result.score:.2f}):\n"
                    f"{result.content[:1000]}..."
                )
        
        full_context = "\n\n".join(context_parts)
        return full_context[:max_length]

    # --------------------------
    # Index Management
    # --------------------------
    
    def build_index(self, embeddings: Dict[str, np.ndarray]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Dictionary of article_id to embedding
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        logger.info("Building FAISS index")
        
        # Convert embeddings to numpy array
        self.article_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[id_] for id_ in self.article_ids]).astype('float32')
        
        # Create and populate index
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embedding_matrix)
        self.index.add(embedding_matrix)
        
        logger.info(f"Built index with {len(self.article_ids)} embeddings")

    def load_index(self, 
                  index_path: str, 
                  metadata_path: Optional[str] = None,
                  content_dir: Optional[str] = None):
        """
        Load FAISS index and optional metadata.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            content_dir: Directory containing article content files
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
            
        # Load index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata if provided
        if metadata_path:
            metadata_path = Path(metadata_path)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
                
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self.article_ids = metadata["article_ids"]
                self.article_metadata = metadata.get("metadata", {})
        
        # Set up content directory
        if content_dir:
            self.content_dir = Path(content_dir)
        
        logger.info(f"Loaded index with {len(self.article_ids)} embeddings")

    def save_index(self, 
                  output_dir: str,
                  metadata: Optional[Dict] = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            output_dir: Directory to save files
            metadata: Additional metadata to store
        """
        if self.index is None:
            raise ValueError("No index to save")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = output_path / f"faiss_index_{self.model_name}.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = output_path / f"faiss_metadata_{self.model_name}.pkl"
        metadata_data = {
            "article_ids": self.article_ids,
            "metadata": metadata or {}
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_data, f)
            
        logger.info(f"Saved index to {index_path}")

    # --------------------------
    # Utility Methods
    # --------------------------
    
    def _get_cache_key(self, text: str) -> str:
        """Generate consistent cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _check_memory(self) -> bool:
        """Check if memory usage is too high."""
        if not self.memory_monitor:
            return False
            
        mem = psutil.virtual_memory()
        return mem.used / mem.total > self.max_memory_usage

    def _cleanup_cache(self):
        """Clean up cache when memory is low."""
        logger.warning("Memory pressure high - cleaning embedding cache")
        self.embedding_cache.clear()
        gc.collect()
        self.current_cache_size = 0

    def _parallel_embed(self, 
                       texts: List[str], 
                       ids: List[str],
                       batch_size: int) -> Dict[str, np.ndarray]:
        """Parallel embedding generation for large datasets."""
        logger.info("Using parallel processing for large dataset")
        
        # Split data into chunks for parallel processing
        num_workers = min(cpu_count(), 8)
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
        """Worker function for parallel embedding."""
        embeddings = self.embed_texts(texts, batch_size=batch_size, show_progress=False)
        return {id_: embedding for id_, embedding in zip(ids, embeddings)}

    def _get_article_content(self, article_id: str) -> Optional[str]:
        """Retrieve article content from disk if available."""
        if not hasattr(self, 'content_dir'):
            return None
            
        content_path = self.content_dir / f"{article_id}.txt"
        if content_path.exists():
            with open(content_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def clear_cache(self):
        """Clear all cached embeddings and free memory."""
        self.embedding_cache.clear()
        gc.collect()
        self.current_cache_size = 0
        logger.info("Cleared embedding cache")

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Alias for embed_texts that matches the expected interface"""
        return self.embed_texts(texts)

# --------------------------
# Lightweight Chatbot Embedder
# --------------------------

class ChatbotEmbedder:
    """
    Lightweight version optimized for real-time chatbot queries.
    Uses precomputed embeddings and FAISS index.
    """
    
    def __init__(self, 
                 index_path: str,
                 metadata_path: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 max_cache_size: int = 1):
        """
        Initialize with precomputed index.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
            model_name: Name of sentence transformer model
            max_cache_size: Maximum cache size in GB
        """
        # Load model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Load index and metadata
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.article_ids = metadata["article_ids"]
            self.article_metadata = metadata.get("metadata", {})
        
        # Initialize cache
        self.embedding_cache = {}
        self.max_cache_size = max_cache_size * (1024**3)
        self.current_cache_size = 0
        
        logger.info(f"Initialized ChatbotEmbedder with {len(self.article_ids)} articles")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query for search."""
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Cache with size check
        self.embedding_cache[cache_key] = embedding
        self.current_cache_size += embedding.nbytes
        
        if self.current_cache_size > self.max_cache_size:
            self.embedding_cache.clear()
            self.current_cache_size = 0
            
        return embedding

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar articles."""
        query_embedding = self.embed_query(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
                
            results.append(SearchResult(
                article_id=self.article_ids[idx],
                score=float(score),
                metadata=self.article_metadata.get(self.article_ids[idx], {})
            ))
            
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        """Get context for LLM response generation."""
        results = self.search(query, top_k=top_k)
        context = []
        
        for result in results:
            context.append(
                f"Article {result.article_id} (relevance: {result.score:.2f}):\n"
                f"{self.article_metadata.get(result.article_id, {}).get('summary', 'No content available')}"
            )
            
        return "\n\n".join(context)

if __name__ == "__main__":
    # Example usage for batch processing
    batch_embedder = ArxivEmbedder()
    
    # Load and process data
    from data_loader import ArxivDataLoader
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data(nrows=10000)
    
    # Generate embeddings
    embeddings = batch_embedder.embed_articles(df)
    
    # Build and save index
    batch_embedder.build_index(embeddings)
    batch_embedder.save_index("data/embeddings/")
    
    # Example usage for chatbot
    chatbot_embedder = ChatbotEmbedder(
        index_path="data/embeddings/faiss_index_all-MiniLM-L6-v2.index",
        metadata_path="data/embeddings/faiss_metadata_all-MiniLM-L6-v2.pkl"
    )
    
    # Process user query
    query = "machine learning applications in healthcare"
    results = chatbot_embedder.search(query)
    context = chatbot_embedder.get_context(query)
    
    print(f"Found {len(results)} relevant articles:")
    for result in results:
        print(f"- {result.article_id} (score: {result.score:.2f})")
    
    print("\nContext for LLM:")
    print(context[:500] + "...")