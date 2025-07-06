#!/usr/bin/env python3
"""
Script pour générer l'index FAISS et les métadonnées pour le chatbot arXiv.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.embedder import ArxivEmbedder
from src.search_engine import ArxivSearchEngine
from src.data_loader import ArxivDataLoader
import logging
import argparse
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_index(data_path: str = "data/processed/articles_clean.csv", 
                  output_dir: str = "data/embeddings/",
                  nrows: int = None):
    """
    Génère l'index FAISS et les métadonnées pour le chatbot.
    
    Args:
        data_path: Chemin vers le fichier CSV des articles nettoyés
        output_dir: Dossier de sortie pour les fichiers d'index
        nrows: Nombre d'articles à traiter (None pour tous)
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
    
    # 2. Générer les embeddings
    logger.info("Génération des embeddings...")
    embedder = ArxivEmbedder()
    embeddings = embedder.embed_articles(df)
    logger.info(f"Généré {len(embeddings)} embeddings")
    
    # 3. Sauvegarder les embeddings
    logger.info("Sauvegarde des embeddings...")
    embedder.save_embeddings(embeddings, output_dir)
    
    # 4. Construire l'index FAISS
    logger.info("Construction de l'index FAISS...")
    search_engine = ArxivSearchEngine()
    search_engine.build_index(embeddings)
    search_engine.load_article_data(df)
    
    # 5. Sauvegarder l'index et les métadonnées
    logger.info("Sauvegarde de l'index et des métadonnées...")
    index_path = os.path.join(output_dir, "arxiv_faiss_index.index")
    metadata_path = os.path.join(output_dir, "arxiv_metadata.pkl")
    
    search_engine.save_index(index_path, metadata_path)
    
    logger.info("=== Index généré avec succès ! ===")
    logger.info(f"Index FAISS: {index_path}")
    logger.info(f"Métadonnées: {metadata_path}")
    logger.info(f"Embeddings: {os.path.join(output_dir, 'embeddings_all-MiniLM-L6-v2.pkl')}")

def generate_quick_index(data_path: str = "data/processed/articles_clean.csv",
                        output_dir: str = "data/embeddings/",
                        nrows: int = 10000):
    """
    Génère un index rapide avec un sous-ensemble d'articles pour les tests.
    
    Args:
        data_path: Chemin vers le fichier CSV des articles nettoyés
        output_dir: Dossier de sortie pour les fichiers d'index
        nrows: Nombre d'articles à traiter (défaut: 10,000)
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
            generate_quick_index(args.data, args.output)
        else:
            generate_index(args.data, args.output, args.nrows)
            
        logger.info("Génération terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        sys.exit(1) 