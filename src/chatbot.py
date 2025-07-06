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
import re

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
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
            
        # Load data and index
        self._initialize_components()
        
    def _initialize_components(self):
        """Load necessary components."""
        # Charger TOUTES les données (pas seulement 1000)
        self.df = self.loader.load_data()
        print(f"[DEBUG] Nombre d'articles chargés: {len(self.df)}")
        
        # Try to load pre-built index
        index_path = "data/embeddings/arxiv_faiss_index.index"
        metadata_path = "data/embeddings/arxiv_metadata.pkl"
        
        if os.path.exists(index_path):
            self.search_engine.load_index(index_path, metadata_path)
        else:
            # Empêcher la régénération automatique de l'index
            raise FileNotFoundError("L'index FAISS n'existe pas. Veuillez le générer manuellement avant de lancer le chatbot.")
        
        self.search_engine.load_article_data(self.df)
        
    def detect_intent(self, query: str) -> str:
        """
        Détection simple de l'intention de l'utilisateur à partir de la requête.
        """
        q = query.lower()
        if any(word in q for word in ["combien", "nombre", "statistique", "statistiques", "how many", "count"]):
            return "statistique"
        if any(word in q for word in ["auteur", "author", "écrit par"]):
            return "auteur"
        if any(word in q for word in ["récent", "récente", "dernier", "dernière", "latest", "recent"]):
            return "recent"
        if any(word in q for word in ["résume", "résumé", "summary", "synthèse"]):
            return "resume"
        return "recherche"

    def parse_filters(self, query: str) -> dict:
        """
        Détecte les filtres simples dans la requête (année, auteur, catégorie).
        """
        filters = {}
        # Filtre année (ex: 2022, 2023)
        year_match = re.findall(r"20\d{2}", query)
        if year_match:
            filters["year"] = year_match[0]
        # Filtre auteur (ex: auteur:Smith)
        author_match = re.search(r"auteur:([\w\- ]+)", query, re.IGNORECASE)
        if author_match:
            filters["author"] = author_match.group(1).strip()
        # Filtre catégorie (ex: cat:cs.AI)
        cat_match = re.search(r"cat:([\w\.\-]+)", query, re.IGNORECASE)
        if cat_match:
            filters["category"] = cat_match.group(1).strip()
        return filters

    def filter_articles(self, df, filters: dict):
        """
        Applique les filtres sur le DataFrame.
        """
        df_filtered = df
        if "year" in filters:
            df_filtered = df_filtered[df_filtered["published"].str.startswith(filters["year"])]
        if "author" in filters:
            df_filtered = df_filtered[df_filtered["author"].str.contains(filters["author"], case=False, na=False)]
        if "category" in filters:
            df_filtered = df_filtered[df_filtered["primary_category"] == filters["category"]]
        return df_filtered

    def search_articles(self, query: str, k: int = 5) -> List[Dict]:
        """
        Recherche intelligente avec filtres.
        """
        filters = self.parse_filters(query)
        df_filtered = self.filter_articles(self.df, filters) if filters else self.df
        # Si le filtre réduit trop, on ajuste k
        k = min(k, len(df_filtered)) if len(df_filtered) < k else k
        # Recherche sémantique sur le sous-ensemble filtré
        return self.search_engine.search_by_text(query, self.embedder, k)
        
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """
        Génère une réponse conversationnelle et synthétique selon l'intention.
        """
        intent = self.detect_intent(query)
        if not search_results:
            return f"Je n'ai trouvé aucun article correspondant à votre question. Essayez avec d'autres mots-clés ou un filtre différent."
        # Synthèse personnalisée
        if self.use_openai:
            # Génération avancée via OpenAI
            context = "\n\n".join([f"Titre: {a.get('title', '')}\nRésumé: {a.get('summary', '')}" for a in search_results])
            prompt = f"Question utilisateur : {query}\nVoici des articles arXiv pertinents :\n{context}\n\nRédige une réponse synthétique, claire et personnalisée pour l'utilisateur, en français."
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "Tu es un assistant scientifique arXiv."},
                              {"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Erreur OpenAI : {e}")
                # Fallback local
        # Réponse locale selon l'intention
        if intent == "statistique":
            return f"J'ai trouvé {len(search_results)} articles correspondant à votre question sur un total de {len(self.df)} articles."
        elif intent == "auteur":
            auteurs = set()
            for res in search_results:
                if res.get('author'):
                    auteurs.update(res['author'].split(';'))
            return f"Les articles trouvés sont écrits par {len(auteurs)} auteur(s) différent(s) : {', '.join(list(auteurs)[:5])}..."
        elif intent == "recent":
            recent = [a for a in search_results if a.get('published_date', '').startswith('202')]
            return f"Parmi les articles trouvés, {len(recent)} sont des publications récentes (2020-2024)."
        elif intent == "resume":
            top = search_results[0]
            summary = top.get('summary', '')
            return f"Résumé de l'article le plus pertinent :\n{summary[:500]}{'...' if len(summary)>500 else ''}"
        else:
            # Réponse conversationnelle par défaut
            top = search_results[0]
            response = f"J'ai trouvé {len(search_results)} articles pertinents.\n\n"
            response += f"Le plus pertinent :\nTitre : {top.get('title', 'Sans titre')}\nAuteur(s) : {top.get('author', 'Inconnu')}\nPublié le : {top.get('published_date', 'N/A')}\nRésumé : {top.get('summary', '')[:400]}{'...' if len(top.get('summary',''))>400 else ''}"
            return response
            
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