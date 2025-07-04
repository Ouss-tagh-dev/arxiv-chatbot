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
        Generate a conversational response based on search results.
        
        Args:
            query: User's query
            search_results: List of article results
            
        Returns:
            Generated response text
        """
        if not self.use_openai:
            return self._simple_response(query, search_results)
            
        # Prepare context for LLM
        context = "\\n\\n".join(
            f"Title: {res['title']}\\nSummary: {res['summary']}"
            for res in search_results
        )
        
        prompt = f"""You are an AI assistant that helps researchers find relevant arXiv papers.
The user asked: "{query}"

Here are some relevant papers:
{context}

Please provide a helpful response summarizing the most relevant papers and how they relate to the query.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._simple_response(query, search_results)
            
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
            return f"I couldn't find any articles matching '{query}'. Please try a different query."
            
        response = [f"I found {len(search_results)} articles related to '{query}':\\n"]
        for i, res in enumerate(search_results, 1):
            response.append(
                f"{i}. {res['title']} (published {res['published_date']})\\n"
                f"   Summary: {res['summary'][:200]}...\\n"
                f"   Link: {res['link']}\\n"
            )
            
        return "\\n".join(response)

if __name__ == "__main__":
    # Example usage
    chatbot = ArxivChatbot()
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        results = chatbot.search_articles(query)
        response = chatbot.generate_response(query, results)
        
        print("\\nChatbot:")
        print(response)
        print("\\n")