"""
Enhanced training script for arXiv chatbot with optimized FAISS indexing,
memory-efficient processing, and parallel execution for large datasets (3.5GB+).
"""

import logging
from pathlib import Path
import time
import gc
import psutil
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse
from data_loader import ArxivDataLoader
from embedder import EnhancedArxivEmbedder as ArxivEmbedder
from search_engine import OptimizedArxivSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitor_memory(max_usage=0.8):
    """Check memory usage and trigger cleanup if needed."""
    mem = psutil.virtual_memory()
    if mem.used / mem.total > max_usage:
        logger.warning(f"Memory usage high ({mem.used/mem.total:.1%}) - performing cleanup")
        gc.collect()
        return True
    return False

def train_embeddings(loader, embedder, output_dir, batch_size=50000):
    """
    Memory-optimized embedding generation with batch processing.
    Returns:
        Tuple of (content_embeddings, title_embeddings)
    """
    logger.info("Starting embedding generation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches to avoid memory issues
    content_embeddings = {}
    title_embeddings = {}
    
    for i, chunk in enumerate(loader.load_iter(batch_size=batch_size)):
        logger.info(f"Processing batch {i+1}")
        
        # Generate content embeddings
        content_batch = embedder.embed_articles(
            chunk, 
            text_field="summary_clean",
            batch_size=1024,
            parallel=True
        )
        content_embeddings.update(content_batch)
        
        # Generate title embeddings
        title_batch = embedder.embed_articles(
            chunk,
            text_field="title_clean",
            batch_size=2048,  # Titles are shorter, can use larger batch
            parallel=True
        )
        title_embeddings.update(title_batch)
        
        # Save intermediate results
        if (i + 1) % 10 == 0:
            embedder.save_embeddings(content_embeddings, output_dir / "content_embeddings")
            embedder.save_embeddings(title_embeddings, output_dir / "title_embeddings")
            logger.info(f"Saved intermediate embeddings after batch {i+1}")
            
            if monitor_memory():
                content_embeddings.clear()
                title_embeddings.clear()
    
    # Final save
    embedder.save_embeddings(content_embeddings, output_dir / "content_embeddings")
    embedder.save_embeddings(title_embeddings, output_dir / "title_embeddings")
    
    return content_embeddings, title_embeddings

def build_faiss_index(search_engine, embeddings, secondary_embeddings=None):
    """Build optimized FAISS index with memory management."""
    logger.info("Building FAISS index")
    
    # Build primary index
    search_engine.build_index(embeddings, secondary_embeddings)
    
    # Optimize index
    if hasattr(search_engine.index, 'make_direct_map'):
        search_engine.index.make_direct_map()
    
    # For large indices, use quantization
    if len(embeddings) > 1_000_000:
        logger.info("Optimizing large index with quantization")
        quantizer = faiss.IndexFlatIP(search_engine.dimension)
        index = faiss.IndexIVFPQ(
            quantizer,
            search_engine.dimension,
            min(4096, len(embeddings) // 1000),  # Number of clusters
            8,  # Number of bits per sub-vector
            8   # Number of sub-vectors
        )
        index.train(np.array(list(embeddings.values())).astype('float32'))
        index.add(np.array(list(embeddings.values())).astype('float32'))
        search_engine.index = index

def train_full_pipeline(args):
    """Full training pipeline with enhanced optimizations."""
    start_time = time.time()
    
    # 1. Initialize components with optimized settings
    logger.info("Initializing components")
    loader = ArxivDataLoader(args.input_data)
    embedder = ArxivEmbedder(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_cache_size=args.max_cache_size
    )
    search_engine = OptimizedArxivSearchEngine(
        index_type=args.index_type,
        use_gpu=args.use_gpu,
        cache_size=args.cache_size
    )

    # 2. Load and clean data in chunks
    logger.info("Loading and cleaning data")
    if not args.skip_cleaning:
        from cleaner import EnhancedArxivDataCleaner
        cleaner = EnhancedArxivDataCleaner(use_spacy=args.use_spacy)
        cleaner.clean_large_dataset(
            input_path=args.input_data,
            output_path=args.clean_output,
            chunk_size=args.chunk_size
        )
        loader = ArxivDataLoader(args.clean_output)

    # 3. Generate embeddings in parallel batches
    content_embeddings, title_embeddings = train_embeddings(
        loader,
        embedder,
        args.embedding_dir,
        batch_size=args.batch_size
    )

    # 4. Build optimized FAISS index
    build_faiss_index(search_engine, content_embeddings, title_embeddings)
    
    # Load article metadata efficiently
    logger.info("Loading article metadata")
    df = loader.load_data(columns=[
        'id', 'title', 'author', 'published_date', 
        'primary_category', 'summary', 'doi'
    ])
    search_engine.load_article_data(df)

    # 5. Save final index and metadata
    logger.info("Saving final index")
    search_engine.save_index(
        args.index_path,
        args.metadata_path,
        args.secondary_index_path
    )

    # 6. Verify index
    logger.info("Verifying index")
    test_query = "machine learning applications in healthcare"
    results = search_engine.search_by_text(test_query, embedder, k=5)
    logger.info(f"Test query results for '{test_query}': {len(results)} articles found")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/3600:.2f} hours")

def parse_args():
    parser = argparse.ArgumentParser(description='Train arXiv chatbot pipeline')
    parser.add_argument('--input-data', default='data/raw/articles.csv',
                      help='Path to input raw data')
    parser.add_argument('--clean-output', default='data/processed/articles_clean.csv',
                      help='Path to save cleaned data')
    parser.add_argument('--embedding-dir', default='data/embeddings',
                      help='Directory to save embeddings')
    parser.add_argument('--index-path', default='data/embeddings/arxiv_faiss_index.index',
                      help='Path to save FAISS index')
    parser.add_argument('--secondary-index-path', default='data/embeddings/arxiv_title_index.index',
                      help='Path to save secondary FAISS index')
    parser.add_argument('--metadata-path', default='data/embeddings/arxiv_metadata.pkl',
                      help='Path to save metadata')
    parser.add_argument('--cache-dir', default='data/cache',
                      help='Directory for caching')
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2',
                      help='Sentence transformer model name')
    parser.add_argument('--index-type', default='HNSW32',
                      choices=['HNSW32', 'IVF4096', 'Flat'],
                      help='FAISS index type')
    parser.add_argument('--batch-size', type=int, default=50000,
                      help='Batch size for embedding generation')
    parser.add_argument('--chunk-size', type=int, default=100000,
                      help='Chunk size for data cleaning')
    parser.add_argument('--max-cache-size', type=int, default=8,
                      help='Max cache size in GB')
    parser.add_argument('--cache-size', type=int, default=10000,
                      help='Max number of queries to cache')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--use-spacy', action='store_true',
                      help='Use spaCy for advanced NLP processing')
    parser.add_argument('--skip-cleaning', action='store_true',
                      help='Skip data cleaning step')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Import FAISS after parsing args to handle GPU setup
        import faiss
        if args.use_gpu:
            faiss.standard_gpu_resources()
        
        train_full_pipeline(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise