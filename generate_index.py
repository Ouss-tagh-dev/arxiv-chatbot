#!/usr/bin/env python3
"""
Script to generate FAISS index and metadata for arXiv chatbot.
Optimized for memory-constrained systems with sequential processing.
"""

import sys
import os
import gc
import numpy as np
import logging
import argparse
from pathlib import Path
import psutil
import pickle
import faiss

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.embedder import ArxivEmbedder
from src.data_loader import ArxivDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_memory(max_usage=0.8):
    """Check if memory usage exceeds threshold"""
    mem = psutil.virtual_memory()
    return mem.used / mem.total > max_usage

def generate_index(data_path: str = "data/processed/articles_clean.csv", 
                 output_dir: str = "data/embeddings/",
                 nrows: int = None, 
                 text_field: str = "summary"):
    """
    Generate FAISS index and metadata for the chatbot.
    Optimized for memory-constrained systems.
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Generating FAISS index ===")
    logger.info(f"Data file: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # 1. Load data
    logger.info("Loading data...")
    loader = ArxivDataLoader(data_path)
    df = loader.load_data(nrows=nrows)
    logger.info(f"Loaded {len(df)} articles")
    
    # 2. Generate embeddings with memory optimization
    logger.info("Generating embeddings...")
    embedder = ArxivEmbedder()
    
    # Memory-efficient processing configuration
    outer_batch_size = 512  # Number of articles to process at once
    inner_batch_size = 32   # Batch size for model inference
    
    # Process in batches
    all_embeddings = {}
    total_batches = (len(df) // outer_batch_size) + 1
    
    for i in range(0, len(df), outer_batch_size):
        batch_df = df.iloc[i:i + outer_batch_size]
        
        logger.info(f"Processing batch {i//outer_batch_size + 1}/{total_batches} ({len(batch_df)} articles)")
        
        # Generate embeddings for this batch
        batch_embeddings = embedder.embed_articles(
            batch_df,
            text_field=text_field,
            batch_size=inner_batch_size,
            parallel=False
        )
        
        # Store results
        all_embeddings.update(batch_embeddings)
        
        # Periodic memory cleanup
        if check_memory(0.7) or (i % (5 * outer_batch_size) == 0 and i > 0):
            logger.info("Performing memory cleanup...")
            embedder.clear_cache()
            gc.collect()
    
    logger.info(f"Generated {len(all_embeddings)} embeddings")
    
    # 3. Build FAISS index
    logger.info("Building FAISS index...")
    
    # Create ordered lists for index building
    article_ids = list(all_embeddings.keys())
    embedding_matrix = np.array([all_embeddings[id_] for id_ in article_ids]).astype('float32')
    
    # Validate embedding matrix
    if embedding_matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape: {embedding_matrix.shape}")
    
    if embedding_matrix.shape[0] == 0:
        raise ValueError("Empty embedding matrix")
    
    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Build index
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embedding_matrix)
    index.add(embedding_matrix)
    
    # Prepare metadata
    metadata = {
        "article_ids": article_ids,
        "metadata": df.set_index("id").to_dict(orient="index")
    }
    
    # 4. Save index and metadata
    logger.info("Saving index and metadata...")
    
    # Save FAISS index
    index_path = output_path / f"faiss_index_{embedder.model_name}.index"
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    metadata_path = output_path / f"faiss_metadata_{embedder.model_name}.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    logger.info("=== Index generated successfully! ===")
    logger.info(f"FAISS index: {index_path}")
    logger.info(f"Metadata: {metadata_path}")

def generate_quick_index(data_path: str = "data/processed/articles_clean.csv",
                       output_dir: str = "data/embeddings/",
                       nrows: int = 10000):
    """
    Generate a quick index with a subset of articles for testing.
    """
    logger.info(f"=== Generating quick index with {nrows} articles ===")
    generate_index(data_path, output_dir, nrows=nrows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FAISS index for arXiv chatbot")
    parser.add_argument(
        "--data",
        default="data/processed/articles_clean.csv",
        help="Path to articles CSV file (default: data/processed/articles_clean.csv)"
    )
    parser.add_argument(
        "--output",
        default="data/embeddings/",
        help="Output directory (default: data/embeddings/)"
    )
    parser.add_argument(
        "--text_field",
        default="summary",
        help="Column name containing text to embed (default: summary)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate quick index with 10,000 articles only"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of articles to process (default: all)"
    )

    args = parser.parse_args()
    
    try:
        if args.quick:
            quick_nrows = args.nrows if args.nrows is not None else 10000
            generate_quick_index(args.data, args.output, quick_nrows)
        else:
            generate_index(args.data, args.output, args.nrows, args.text_field)
            
        logger.info("Index generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        sys.exit(1)