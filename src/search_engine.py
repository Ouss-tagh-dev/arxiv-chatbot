"""
Module for indexing and searching arXiv articles using FAISS.
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple
import logging
import pickle
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivSearchEngine:
    def __init__(self):
        """Initialize the search engine."""
        self.index = None
        self.article_ids = []
        self.article_data = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
    def build_index(self, embeddings: Dict[str, np.ndarray]):
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Dictionary mapping article IDs to embeddings
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        # Convert embeddings to numpy array
        self.article_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[id_] for id_ in self.article_ids]).astype('float32')
        
        # Create and train index
        self.dimension = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embedding_matrix)
        
        logger.info(f"Built FAISS index with {len(self.article_ids)} embeddings")
        
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load a pre-built FAISS index from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to article metadata file
        """
        self.index = faiss.read_index(index_path)
        
        # Load article metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.article_ids = metadata["article_ids"]
            self.article_data = metadata["article_data"]
            
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index with {len(self.article_ids)} embeddings")
        
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if not self.index:
            raise ValueError("Index not built")
            
        faiss.write_index(self.index, index_path)
        
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
        Load article metadata for search results.
        
        Args:
            df: DataFrame containing article metadata
        """
        self.article_data = df.set_index("id").to_dict("index")
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar articles.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            List of result dictionaries with article metadata
        """
        if not self.index:
            raise ValueError("Index not built")
            
        if not self.article_data:
            raise ValueError("Article data not loaded")
            
        # Convert query embedding to numpy array
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i < 0:  # FAISS returns -1 for invalid indices
                continue
                
            article_id = self.article_ids[i]
            article_info = self.article_data.get(article_id, {})
            article_info["similarity"] = float(distance)
            results.append(article_info)
            
        return results
        
    def search_by_text(self, query_text: str, embedder, k: int = 5) -> List[Dict]:
        """
        Search using raw text query.
        
        Args:
            query_text: Text query to search for
            embedder: ArxivEmbedder instance
            k: Number of results to return
            
        Returns:
            List of result dictionaries with article metadata
        """
        # Embed the query
        query_embedding = embedder.generate_embeddings([query_text])[0]
        
        # Search using the embedding
        return self.search(query_embedding, k)

if __name__ == "__main__":
    # Example usage
    from embedder import ArxivEmbedder
    from data_loader import ArxivDataLoader
    
    # Load data
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data(nrows=1000)
    
    # Generate embeddings
    embedder = ArxivEmbedder()
    embeddings = embedder.embed_articles(df)
    
    # Build search index
    search_engine = ArxivSearchEngine()
    search_engine.build_index(embeddings)
    search_engine.load_article_data(df)
    
    # Perform a search
    query = "machine learning applications in healthcare"
    results = search_engine.search_by_text(query, embedder)
    
    print(f"Search results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")