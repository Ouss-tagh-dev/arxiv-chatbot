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
import pandas as pd

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
    
    # Clean the data - remove rows with missing critical information
    df = df.dropna(subset=['published_date', 'primary_category'])
    
    # Ensure published_date is properly parsed
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df = df.dropna(subset=['published_date'])  # Remove rows where date parsing failed
    
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
    
    try:
        # Load components
        embedder, search_engine, df = load_components()
        
        # Check if we have any data after cleaning
        if df.empty:
            st.error("No valid data available. Please check your data files.")
            return
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Get year range with proper handling of NaN values
        min_year = int(df["published_date"].dt.year.min())
        max_year = int(df["published_date"].dt.year.max())
        
        year_range = st.sidebar.slider(
            "Publication Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(max(2010, min_year), max_year)
        )
        
        # Get available categories
        categories = sorted(df["primary_category"].unique())
        
        # Default categories - use available ones if the defaults don't exist
        default_categories = ["cs.CL", "cs.LG", "cs.AI"]
        available_defaults = [cat for cat in default_categories if cat in categories]
        if not available_defaults:
            available_defaults = categories[:3]  # Use first 3 categories if defaults not available
        
        selected_categories = st.sidebar.multiselect(
            "Categories",
            categories,
            default=available_defaults
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
                    (df["published_date"].dt.year <= year_range[1])
                ]
                
                # Apply category filter if categories selected
                if selected_categories:
                    filtered_df = filtered_df[filtered_df["primary_category"].isin(selected_categories)]
                
                # Check if we have results after filtering
                if filtered_df.empty:
                    st.warning("No articles found matching your filters. Try broadening your search criteria.")
                    return
                
                # Update search engine with filtered data
                search_engine.load_article_data(filtered_df)
                
                # Perform search
                results = search_engine.search_by_text(query, embedder, k=num_results)
                
                # Display results
                if results:
                    st.subheader(f"Search Results ({len(results)} found)")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result.get('title', 'No Title')} (Similarity: {result.get('similarity', 0):.3f})"):
                            st.markdown(f"**Authors**: {result.get('author', 'Unknown')}")
                            
                            # Format publication date
                            pub_date = result.get('published_date', 'Unknown')
                            if pd.notna(pub_date):
                                if isinstance(pub_date, str):
                                    pub_date = pd.to_datetime(pub_date).strftime('%Y-%m-%d')
                                else:
                                    pub_date = pub_date.strftime('%Y-%m-%d')
                            
                            st.markdown(f"**Published**: {pub_date}")
                            st.markdown(f"**Categories**: {result.get('category', 'Unknown')}")
                            
                            # Display summary
                            summary = result.get('summary', 'No summary available')
                            if len(summary) > 500:
                                summary = summary[:500] + "..."
                            st.markdown(f"**Summary**: {summary}")
                            
                            # Create arXiv link
                            arxiv_id = result.get('id', '')
                            if arxiv_id:
                                st.markdown(f"[arXiv Link](https://arxiv.org/abs/{arxiv_id})")
                            
                            # Add DOI link if available
                            doi = result.get('doi')
                            if doi and pd.notna(doi):
                                st.markdown(f"[DOI](https://doi.org/{doi})")
                else:
                    st.info("No results found for your query. Try different keywords or adjust your filters.")
    
    except FileNotFoundError as e:
        st.error(f"Required files not found: {e}")
        st.info("Please ensure you have run the data preprocessing and embedding generation steps.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check your data files and try again.")

if __name__ == "__main__":
    main()