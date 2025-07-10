#!/usr/bin/env python3
"""
Script pour générer l'index FAISS et les métadonnées pour le chatbot arXiv.
Optimized for 16GB RAM systems with sequential processing and memory management.
"""

import sys
import os
import gc
import numpy as np  # Added missing import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.embedder import EnhancedArxivEmbedder as ArxivEmbedder
from src.search_engine import OptimizedArxivSearchEngine as ArxivSearchEngine
from src.data_loader import ArxivDataLoader
import logging
import argparse
from pathlib import Path
import psutil

# Configuration du logging
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
    Génère l'index FAISS et les métadonnées pour le chatbot.
    Optimized for memory-constrained systems.
    """
    
    # Créer le dossier de sortie
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Génération de l'index FAISS ===")
    logger.info(f"Fichier de données: {data_path}")
    logger.info(f"Dossier de sortie: {output_dir}")
    
    # 1. Charger les données
    logger.info("Chargement des données...")
    loader = ArxivDataLoader(data_path)
    df = loader.load_data(nrows=nrows)
    logger.info(f"Chargé {len(df)} articles")
    
    # 2. Générer les embeddings avec optimisation mémoire
    logger.info("Génération des embeddings...")
    embedder = ArxivEmbedder()
    
    # Configuration pour économiser la mémoire
    outer_batch_size = 512  # Number of articles to process at once
    inner_batch_size = 32   # Batch size for model inference
    texts = df[text_field].fillna("").astype(str).tolist()
    ids = df["id"].astype(str).tolist()
    
    embeddings = {}
    total_batches = (len(texts) // outer_batch_size) + 1
    
    for i in range(0, len(texts), outer_batch_size):
        batch_texts = texts[i:i + outer_batch_size]
        batch_ids = ids[i:i + outer_batch_size]
        
        logger.info(f"Traitement du lot {i//outer_batch_size + 1}/{total_batches} ({len(batch_texts)} articles)")
        
        # Generate embeddings with small batches
        batch_embeddings = embedder.model.encode(
            batch_texts,
            batch_size=inner_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Store results
        for id_, embedding in zip(batch_ids, batch_embeddings):
            embeddings[id_] = embedding
        
        # Nettoyage mémoire périodique
        if check_memory(0.7) or (i % (5 * outer_batch_size) == 0 and i > 0):
            logger.info("Nettoyage mémoire...")
            embedder.clear_cache()
            gc.collect()
    
    logger.info(f"Généré {len(embeddings)} embeddings")
    
    # 3. Sauvegarder les embeddings
    logger.info("Sauvegarde des embeddings...")
    embedder.save_embeddings(embeddings, output_dir)
    
    # 4. Construire l'index FAISS
    logger.info("Construction de l'index FAISS...")
    search_engine = ArxivSearchEngine()
    
    # Create article IDs list to match the order of embeddings
    article_ids = list(embeddings.keys())
    
    # Convert embeddings to numpy array in the same order as article_ids
    embedding_matrix = np.array([embeddings[id_] for id_ in article_ids], dtype=np.float32)
    
    # Validate embedding matrix
    if embedding_matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape: {embedding_matrix.shape}")
    
    if embedding_matrix.shape[0] == 0:
        raise ValueError("Empty embedding matrix")
    
    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Set article IDs in search engine before building index
    search_engine.article_ids = article_ids
    
    # Build index with error handling
    try:
        search_engine.build_index(embedding_matrix)
        search_engine.load_article_data(df)
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        logger.error(f"Embedding matrix shape: {embedding_matrix.shape}")
        logger.error(f"Embedding matrix dtype: {embedding_matrix.dtype}")
        raise
    
    # 5. Sauvegarder l'index et les métadonnées
    logger.info("Sauvegarde de l'index et des métadonnées...")
    index_path = os.path.join(output_dir, "arxiv_faiss_index.index")
    metadata_path = os.path.join(output_dir, "arxiv_metadata.pkl")
    
    search_engine.save_index(index_path, metadata_path)
    
    logger.info("=== Index généré avec succès ! ===")
    logger.info(f"Index FAISS: {index_path}")
    logger.info(f"Métadonnées: {metadata_path}")

def generate_quick_index(data_path: str = "data/processed/articles_clean.csv",
                       output_dir: str = "data/embeddings/",
                       nrows: int = 10000):  # Default to 10,000 for quick mode
    """
    Génère un index rapide avec un sous-ensemble d'articles pour les tests.
    """
    logger.info(f"=== Génération d'un index rapide avec {nrows} articles ===")
    generate_index(data_path, output_dir, nrows=nrows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération de l'index FAISS pour le chatbot arXiv")
    parser.add_argument(
        "--data",
        default="data/processed/articles_clean.csv",
        help="Chemin vers le fichier CSV des articles (défaut: data/processed/articles_clean.csv)"
    )
    parser.add_argument(
        "--output",
        default="data/embeddings/",
        help="Dossier de sortie (défaut: data/embeddings/)"
    )
    parser.add_argument(
        "--text_field",
        default="summary",
        help="Column name containing text to embed (default: summary)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Générer un index rapide avec 10,000 articles seulement"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Nombre d'articles à traiter (défaut: tous)"
    )

    args = parser.parse_args()
    
    try:
        if args.quick:
            quick_nrows = args.nrows if args.nrows is not None else 10000
            generate_quick_index(args.data, args.output, quick_nrows)
        else:
            generate_index(args.data, args.output, args.nrows, args.text_field)
            
        logger.info("Génération terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        sys.exit(1)