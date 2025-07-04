"""
Module for generating semantic embeddings of arXiv articles using sentence transformers.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    
    def embed_articles(self, df: pd.DataFrame, text_field: str = "summary") -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all articles in a DataFrame.
        
        Args:
            df: DataFrame containing articles
            text_field: Column name containing text to embed
            
        Returns:
            Dictionary mapping article IDs to embeddings
        """
        embeddings = {}
        texts = df[text_field].tolist()
        ids = df["id"].tolist()
        
        # Generate embeddings in batches
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = self.generate_embeddings(batch_texts)
            
            for id_, embedding in zip(batch_ids, batch_embeddings):
                embeddings[id_] = embedding
                
        logger.info(f"Generated embeddings for {len(embeddings)} articles")
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_dir: str):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Dictionary of embeddings
            output_dir: Directory to save embeddings
        """
        output_path = Path(output_dir) / f"embeddings_{self.model_name}.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
            
        logger.info(f"Saved embeddings to {output_path}")
        
    def load_embeddings(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to embeddings file
            
        Returns:
            Dictionary of embeddings
        """
        with open(input_path, "rb") as f:
            embeddings = pickle.load(f)
            
        logger.info(f"Loaded embeddings from {input_path}")
        return embeddings

if __name__ == "__main__":
    # Example usage
    from data_loader import ArxivDataLoader
    
    # Load sample data
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data(nrows=1000)
    
    # Generate embeddings
    embedder = ArxivEmbedder()
    embeddings = embedder.embed_articles(df)
    
    # Save embeddings
    embedder.save_embeddings(embeddings, "data/embeddings/")