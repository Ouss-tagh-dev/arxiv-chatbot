import sys
import os
# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
import argparse
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict, Optional, Tuple
import streamlit as st
import psutil
import time

# Import the optimized search engine
from search_engine import OptimizedArxivSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArxivEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

class ArxivChatbot:
    def __init__(self, data_path: str = "data/processed/articles_clean.csv"):
        self.embedder = ArxivEmbedder()
        # Use the optimized search engine instead
        self.search_engine = OptimizedArxivSearchEngine()
        self.load_data(data_path)
        
    def load_data(self, data_path: str):
        try:
            # Load pre-built index
            index_path = "data/embeddings/arxiv_faiss_index.index"
            metadata_path = "data/embeddings/arxiv_metadata.pkl"
            
            # Check if files exist
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            # Load the index using the optimized search engine
            self.search_engine.load_index(index_path, metadata_path)
            
            # Load article data if CSV exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, nrows=20000)  # Match your nrows
                self.search_engine.load_article_data(df)
            else:
                logger.warning(f"CSV file not found: {data_path}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Fallback: create empty search engine
            self.search_engine = OptimizedArxivSearchEngine()
            raise
        
    def generate_embeddings(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        texts = df['summary'].fillna("").astype(str).tolist()
        ids = df["id"].astype(str).tolist()
        embeddings = self.embedder.generate_embeddings(texts)
        return {id_: embedding for id_, embedding in zip(ids, embeddings)}
        
    def search_articles(self, query: str, k: int = 5) -> List[Dict]:
        try:
            # Use the optimized search engine's search_by_text method
            return self.search_engine.search_by_text(query, self.embedder, k)
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []
        
    def generate_response(self, query: str, results: List[Dict]) -> str:
        if not results:
            return "No relevant articles found."
            
        response = f"Found {len(results)} articles about '{query}':\n\n"
        for i, res in enumerate(results[:3], 1):
            response += f"{i}. {res.get('title', 'N/A')}\n"
            response += f"   Authors: {res.get('author', 'N/A')}\n"
            response += f"   Published: {res.get('published_date', 'N/A')}\n"
            response += f"   Category: {res.get('primary_category', 'N/A')}\n"
            response += f"   Similarity: {res.get('similarity', 0):.3f}\n\n"
            
        return response

def run_streamlit():
    st.set_page_config(page_title="arXiv Chatbot", layout="wide")
    st.title("arXiv Research Assistant")
    
    # Initialize chatbot with error handling
    if "chatbot" not in st.session_state:
        try:
            with st.spinner("Loading chatbot..."):
                st.session_state.chatbot = ArxivChatbot()
                st.success("Chatbot loaded successfully!")
        except Exception as e:
            st.error(f"Error loading chatbot: {e}")
            st.info("Please ensure that the index files exist in the data/embeddings/ directory.")
            return
    
    # Chat interface
    if prompt := st.chat_input("Ask about arXiv research..."):
        with st.spinner("Searching..."):
            try:
                results = st.session_state.chatbot.search_articles(prompt)
                response = st.session_state.chatbot.generate_response(prompt, results)
                
                st.chat_message("user").write(prompt)
                st.chat_message("assistant").write(response)
                
                # Show detailed results
                if results:
                    with st.expander("Search Results Details"):
                        for i, res in enumerate(results, 1):
                            st.write(f"**{i}. {res.get('title', 'N/A')}**")
                            st.write(f"- Authors: {res.get('author', 'N/A')}")
                            st.write(f"- Published: {res.get('published_date', 'N/A')}")
                            st.write(f"- Category: {res.get('primary_category', 'N/A')}")
                            st.write(f"- Similarity Score: {res.get('similarity', 0):.3f}")
                            if res.get('summary'):
                                st.write(f"- Summary: {res.get('summary', '')[:200]}...")
                            st.write("---")
                            
            except Exception as e:
                st.error(f"Error during search: {e}")

def run_cli():
    try:
        chatbot = ArxivChatbot()
        print("arXiv Chatbot - Type 'quit' to exit")
        
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ["quit", "exit"]:
                break
                
            results = chatbot.search_articles(query)
            print(chatbot.generate_response(query, results))
            
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Please ensure that the index files exist in the data/embeddings/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arXiv Chatbot")
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="streamlit")
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        run_streamlit()
    else:
        run_cli()