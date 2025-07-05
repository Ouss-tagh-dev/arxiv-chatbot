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
from chatbot import ArxivChatbot

# Configure page
st.set_page_config(
    page_title="arXiv Chatbot",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS for chat bubbles and modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .database-info {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_chatbot():
    """Load the chatbot with caching."""
    return ArxivChatbot()

def main():
    st.markdown('<h1 class="main-header">arXiv Chatbot</h1>', unsafe_allow_html=True)

    # Simulate database info (since only arXiv CSV is used)
    try:
        loader = ArxivDataLoader("data/processed/articles_clean.csv")
        df = loader.load_data()
        total_articles = len(df)
        db_status = 'Connecté' if total_articles > 0 else 'Non connecté'
    except Exception:
        total_articles = 0
        db_status = 'Erreur'

    st.markdown(f'''
    <div class="database-info">
    <strong>✅ Base de données arXiv :</strong><br>
    - Statut : {db_status} ({total_articles} articles)<br>
    </div>
    ''', unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("## Paramètres")
        num_results = st.slider(
            "Nombre max de résultats",
            min_value=1,
            max_value=20,
            value=5,
            help="Nombre maximum d'articles à afficher"
        )
        st.markdown("### Filtres")
        # Year filter
        author_filter = st.text_input(
            "Auteur (nom ou partie du nom)",
            placeholder="Ex: Smith, John, etc."
        )
        if total_articles > 0:
            min_year = int(df["published_date"].dt.year.min())
            max_year = int(df["published_date"].dt.year.max())
            year_filter = st.selectbox(
                "Année",
                options=["Toutes"] + [str(y) for y in range(max_year, min_year-1, -1)],
                help="Filtrer par année de publication"
            )
            # Fix category extraction: get all unique, non-null, non-'Unknown' categories
            categories = sorted([c for c in df["primary_category"].dropna().unique() if c and c != 'Unknown'])
            selected_categories = st.multiselect(
                "Catégories",
                categories,
                default=categories[:3] if len(categories) >= 3 else categories
            )
        st.markdown("---")
        st.markdown("## Aide")
        st.markdown("""
        **Exemples de questions :**
        - Articles on machine learning
        - Combien d'articles sur l'IA ?
        - Articles récents sur deep learning
        - Auteurs spécialisés en NLP
        - Tendances en computer vision
        """)

    # --- CHAT INTERFACE ---
    st.markdown("## Conversation avec l'assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom
    if prompt := st.chat_input("Posez votre question sur arXiv..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # --- SEARCH LOGIC ---
        try:
            embedder, search_engine, df = load_components()
            chatbot = load_chatbot()
            
            filtered_df = df.copy()
            # Apply year filter
            if total_articles > 0 and year_filter != "Toutes":
                filtered_df = filtered_df[filtered_df["published_date"].dt.year == int(year_filter)]
            # Apply category filter
            if total_articles > 0 and selected_categories:
                filtered_df = filtered_df[filtered_df["primary_category"].isin(selected_categories)]
            # Apply author filter (case-insensitive substring match)
            if total_articles > 0 and author_filter.strip():
                filtered_df = filtered_df[filtered_df["author"].str.contains(author_filter, case=False, na=False)]
            
            search_engine.load_article_data(filtered_df)
            results = search_engine.search_by_text(prompt, embedder, k=num_results)
            
            # Generate intelligent response using chatbot
            response = chatbot.generate_response(prompt, results)
            
            # Display the intelligent response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display detailed results in expandable sections
            if results:
                st.markdown("### 📚 Articles détaillés")
                for i, r in enumerate(results):
                    with st.expander(f"{i+1}. {r.get('title', 'No Title')} (Sim: {r.get('similarity', 0):.3f})"):
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown(f"**Auteurs**: {r.get('author', 'Unknown')}")
                        pub_date = r.get('published_date', 'Unknown')
                        if pd.notna(pub_date):
                            if isinstance(pub_date, str):
                                pub_date = pd.to_datetime(pub_date).strftime('%Y-%m-%d')
                            else:
                                pub_date = pub_date.strftime('%Y-%m-%d')
                        st.markdown(f"**Publié**: {pub_date}")
                        # Show correct category
                        cat = r.get('category', r.get('primary_category', ''))
                        if not cat or cat == 'Unknown':
                            cat = r.get('primary_category', '')
                        st.markdown(f"**Catégories**: {cat if cat else 'Non spécifié'}")
                        summary = r.get('summary', 'No summary available')
                        if len(summary) > 500:
                            summary = summary[:500] + "..."
                        st.markdown(f"**Résumé**: {summary}")
                        arxiv_id = r.get('id', '')
                        if arxiv_id:
                            st.markdown(f"[Lien arXiv](https://arxiv.org/abs/{arxiv_id})")
                        doi = r.get('doi')
                        if doi and pd.notna(doi):
                            st.markdown(f"[DOI](https://doi.org/{doi})")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("### ❌ Aucun article trouvé")
                st.info("Essayez avec des mots-clés différents ou une question plus générale.")
                
        except Exception as e:
            error_msg = f"Erreur lors de la recherche : {e}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # --- TABS FOR ANALYTICS/ADVANCED SEARCH ---
    tab1, tab2 = st.tabs(["Analyses", "Recherche avancée"])
    with tab1:
        st.markdown("_Fonctionnalités d'analyse à venir..._")
    with tab2:
        st.markdown("_Recherche avancée à venir..._")

if __name__ == "__main__":
    main()