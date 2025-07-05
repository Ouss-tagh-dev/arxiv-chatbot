"""
Enhanced conversational interface for the arXiv chatbot.
"""

from search_engine import ArxivSearchEngine
from embedder import ArxivEmbedder
from data_loader import ArxivDataLoader
import openai
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivChatbot:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the arXiv chatbot.
        
        Args:
            openai_api_key: OpenAI API key for conversational responses
        """
        self.embedder = ArxivEmbedder()
        self.search_engine = ArxivSearchEngine()
        self.loader = ArxivDataLoader("data/processed/articles_clean.csv")
        
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
            
        # Load data and index
        self._initialize_components()
        
    def _initialize_components(self):
        """Load necessary components."""
        self.df = self.loader.load_data()
        
        # Try to load pre-built index
        index_path = "data/embeddings/arxiv_faiss_index.index"
        metadata_path = "data/embeddings/arxiv_metadata.pkl"
        
        if os.path.exists(index_path):
            self.search_engine.load_index(index_path, metadata_path)
        else:
            # Generate embeddings and build index
            embeddings = self.embedder.embed_articles(self.df)
            self.search_engine.build_index(embeddings)
            self.search_engine.save_index(index_path, metadata_path)
            
        self.search_engine.load_article_data(self.df)
        
    def search_articles(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search arXiv articles with a natural language query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of article results with metadata
        """
        return self.search_engine.search_by_text(query, self.embedder, k)
        
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """
        Generate an intelligent response based on search results and dataset analysis.
        
        Args:
            query: User's query
            search_results: List of article results
            
        Returns:
            Generated response text
        """
        if not search_results:
            return f"Je n'ai trouvé aucun article correspondant à '{query}'. Essayez avec des mots-clés différents ou une question plus générale."
        
        # Analyze the dataset for context
        total_articles = len(self.df)
        query_lower = query.lower()
        
        # Generate intelligent response based on search results
        response_parts = []
        
        # Add contextual information based on query type
        if any(word in query_lower for word in ['combien', 'nombre', 'count', 'how many']):
            response_parts.append(f"Dans notre base de données de {total_articles} articles ArXiv, j'ai trouvé {len(search_results)} articles pertinents pour votre question.")
        
        elif any(word in query_lower for word in ['récent', 'récente', 'recent', 'latest', 'dernière']):
            recent_count = sum(1 for res in search_results if res.get('published_date', '').startswith('202'))
            response_parts.append(f"Parmi les articles trouvés, {recent_count} sont des publications récentes (2020-2024).")
        
        elif any(word in query_lower for word in ['auteur', 'author', 'écrit par']):
            authors = set()
            for res in search_results:
                if res.get('author'):
                    authors.update(res['author'].split(';'))
            response_parts.append(f"Les articles trouvés sont écrits par {len(authors)} auteurs différents.")
        
        else:
            response_parts.append(f"J'ai trouvé {len(search_results)} articles pertinents dans notre base de données ArXiv.")
        
        # Add summary of top results
        if search_results:
            top_result = search_results[0]
            response_parts.append(f"\n📄 **Article le plus pertinent :** {top_result.get('title', 'Sans titre')}")
            
            if top_result.get('summary'):
                summary = top_result['summary'][:300]
                if len(top_result['summary']) > 300:
                    summary += "..."
                response_parts.append(f"\n📝 **Résumé :** {summary}")
            
            if top_result.get('author'):
                response_parts.append(f"\n👤 **Auteur(s) :** {top_result['author']}")
            
            if top_result.get('published_date'):
                response_parts.append(f"\n📅 **Publié le :** {top_result['published_date']}")
        
        # Add category information if available
        categories = set()
        for res in search_results:
            if res.get('primary_category'):
                categories.add(res['primary_category'])
        
        if categories:
            response_parts.append(f"\n🏷️ **Catégories principales :** {', '.join(categories)}")
        
        return "\n".join(response_parts)
            
    def _simple_response(self, query: str, search_results: List[Dict]) -> str:
        """
        Fallback response generator without LLM.
        
        Args:
            query: User's query
            search_results: List of article results
            
        Returns:
            Simple response text
        """
        if not search_results:
            return f"Je n'ai trouvé aucun article correspondant à '{query}'. Essayez avec des mots-clés différents."
            
        response = [f"J'ai trouvé {len(search_results)} articles liés à '{query}':\n"]
        for i, res in enumerate(search_results, 1):
            response.append(
                f"{i}. {res['title']} (publié le {res['published_date']})\n"
                f"   Résumé: {res['summary'][:200]}...\n"
                f"   Lien: {res['link']}\n"
            )
            
        return "\n".join(response)

if __name__ == "__main__":
    # Example usage
    chatbot = ArxivChatbot()
    
    while True:
        query = input("Vous: ")
        if query.lower() in ["exit", "quit", "quitter"]:
            break
            
        results = chatbot.search_articles(query)
        response = chatbot.generate_response(query, results)
        
        print("\nChatbot:")
        print(response)
        print("\n")