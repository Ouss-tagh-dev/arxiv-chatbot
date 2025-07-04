"""
Streamlit interface for the arXiv chatbot.
"""

import streamlit as st
from search_engine import ArxivSearchEngine
from embedder import ArxivEmbedder
from data_loader import ArxivDataLoader
import pickle
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="arXiv Chatbot",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_components():
    """Load the search engine and embedder with caching."""
    # Load data
    loader = ArxivDataLoader("data/processed/articles_clean.csv")
    df = loader.load_data()
    
    # Load embeddings and index
    embedder = ArxivEmbedder()
    search_engine = ArxivSearchEngine()
    
    embeddings_path = "data/embeddings/embeddings_all-MiniLM-L6-v2.pkl"
    index_path = "data/embeddings/arxiv_faiss_index.index"
    metadata_path = "data/embeddings/arxiv_metadata.pkl"
    
    if os.path.exists(index_path):
        search_engine.load_index(index_path, metadata_path)
    else:
        embeddings = embedder.load_embeddings(embeddings_path)
        search_engine.build_index(embeddings)
        search_engine.save_index(index_path, metadata_path)
        
    search_engine.load_article_data(df)
    
    return embedder, search_engine, df

def main():
    st.title("arXiv Semantic Search Chatbot")
    st.markdown("Search through arXiv articles using natural language queries.")
    
    # Load components
    embedder, search_engine, df = load_components()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    year_range = st.sidebar.slider(
        "Publication Year Range",
        min_value=int(df["published_date"].dt.year.min()),
        max_value=int(df["published_date"].dt.year.max()),
        value=(2010, 2025)
    )
    
    categories = df["primary_category"].unique()
    selected_categories = st.sidebar.multiselect(
        "Categories",
        categories,
        default=["cs.CL", "cs.LG", "cs.AI"]
    )
    
    # Main search interface
    query = st.text_input(
        "Search arXiv articles",
        placeholder="e.g. 'Recent advances in transformer models'"
    )
    
    num_results = st.slider("Number of results", 1, 20, 5)
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            # Apply filters
            filtered_df = df[
                (df["published_date"].dt.year >= year_range[0]) & 
                (df["published_date"].dt.year <= year_range[1]) & 
                (df["primary_category"].isin(selected_categories))
            ]
            search_engine.load_article_data(filtered_df)
            
            # Perform search
            results = search_engine.search_by_text(query, embedder, k=num_results)
            
            # Display results
            st.subheader("Search Results")
            for i, result in enumerate(results, 1):
                with st.expander(f"{i}. {result['title']} (Similarity: {result['similarity']:.3f})"):
                    st.markdown(f"**Authors**: {result['author']}")
                    st.markdown(f"**Published**: {result['published_date']}")
                    st.markdown(f"**Categories**: {result['category']}")
                    st.markdown(f"**Summary**: {result['summary']}")
                    st.markdown(f"[arXiv Link]({result['link']})")
                    
                    if result.get("doi"):
                        st.markdown(f"[DOI](https://doi.org/{result['doi']})")

if __name__ == "__main__":
    main()