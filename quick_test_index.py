#!/usr/bin/env python3
"""
Script pour générer un index de test très rapide avec 1000 articles seulement.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.embedder import ArxivEmbedder
from src.search_engine import ArxivSearchEngine
from src.data_loader import ArxivDataLoader
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_index():
    """Génère un index de test avec 1000 articles seulement."""
    
    logger.info("=== Génération d'un index de test rapide ===")
    
    # Charger seulement 1000 articles
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data(nrows=1000)
    logger.info(f"Chargé {len(df)} articles pour le test")
    
    # Générer les embeddings
    embedder = ArxivEmbedder()
    embeddings = embedder.embed_articles(df)
    logger.info(f"Généré {len(embeddings)} embeddings")
    
    # Construire l'index FAISS
    search_engine = ArxivSearchEngine()
    search_engine.build_index(embeddings)
    search_engine.load_article_data(df)
    
    # Sauvegarder
    output_dir = "data/embeddings/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    index_path = os.path.join(output_dir, "arxiv_faiss_index.index")
    metadata_path = os.path.join(output_dir, "arxiv_metadata.pkl")
    
    search_engine.save_index(index_path, metadata_path)
    
    logger.info("=== Index de test généré avec succès ! ===")
    logger.info(f"Index FAISS: {index_path}")
    logger.info(f"Métadonnées: {metadata_path}")

if __name__ == "__main__":
    try:
        generate_test_index()
        logger.info("Génération terminée avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        sys.exit(1) 