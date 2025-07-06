"""
Streamlit interface for the arXiv chatbot.
"""

import streamlit as st
from chatbot import ArxivChatbot
import time
import pandas as pd

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

@st.cache_resource(ttl=None)
def initialize_chatbot():
    """
    Initialize chatbot ONCE and cache it permanently.
    This function will only run once per session.
    """
    with st.spinner("🔄 Initialisation du chatbot et chargement des données..."):
        start_time = time.time()
        
        # Create chatbot instance
        chatbot = ArxivChatbot()
        
        # Calculate loading time
        load_time = time.time() - start_time
        
        # Store initialization info in session state
        st.session_state.chatbot_ready = True
        st.session_state.load_time = load_time
        st.session_state.total_articles = len(chatbot.df)
        
        st.success(f"✅ Chatbot initialisé en {load_time:.2f} secondes")
        return chatbot

def main():
    st.markdown('<h1 class="main-header">arXiv Chatbot</h1>', unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot_ready" not in st.session_state:
        st.session_state.chatbot_ready = False
    if "total_articles" not in st.session_state:
        st.session_state.total_articles = 0

    # Load chatbot (will be cached after first load)
    chatbot = initialize_chatbot()
    
    # Display database info
    total_articles = len(chatbot.df)
    db_status = 'Connecté' if total_articles > 0 else 'Non connecté'

    st.markdown(f'''
    <div class="database-info">
    <strong>✅ Base de données arXiv :</strong><br>
    - Statut : {db_status} ({total_articles:,} articles)<br>
    - Mode complet avec cache optimisé
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
            # Fix date parsing for year filter
            try:
                # Ensure published_date is datetime
                if 'published_date' in chatbot.df.columns:
                    if chatbot.df['published_date'].dtype == 'object':
                        chatbot.df['published_date'] = pd.to_datetime(chatbot.df['published_date'], errors='coerce')
                    
                    min_year = int(chatbot.df["published_date"].dt.year.min())
                    max_year = int(chatbot.df["published_date"].dt.year.max())
                    year_filter = st.selectbox(
                        "Année",
                        options=["Toutes"] + [str(y) for y in range(max_year, min_year-1, -1)],
                        help="Filtrer par année de publication"
                    )
                else:
                    year_filter = "Toutes"
            except:
                year_filter = "Toutes"
            
            # Fix category extraction: get all unique, non-null, non-'Unknown' categories
            categories = sorted([c for c in chatbot.df["primary_category"].dropna().unique() if c and c != 'Unknown'])
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
        - 2023 cat:cs.AI résumé
        """)

    # --- CHAT INTERFACE ---
    st.markdown("## Conversation avec l'assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom
    if prompt := st.chat_input("Posez votre question sur arXiv..."):
        # Only process if chatbot is ready
        if not st.session_state.chatbot_ready:
            st.error("⏳ Veuillez attendre que le chatbot soit initialisé...")
            return
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # --- SEARCH LOGIC ---
        try:
            with st.spinner("🔍 Recherche en cours..."):
                # Apply filters to the query if needed
                enhanced_query = prompt
                
                # Add year filter to query if selected
                if year_filter != "Toutes":
                    enhanced_query += f" {year_filter}"
                
                # Add category filter to query if selected
                if selected_categories:
                    for cat in selected_categories:
                        enhanced_query += f" cat:{cat}"
                
                # Add author filter to query if provided
                if author_filter.strip():
                    enhanced_query += f" auteur:{author_filter}"
                
                # Search using the chatbot (which handles all the logic)
                results = chatbot.search_articles(enhanced_query, k=num_results)
                
                # Generate intelligent response using chatbot
                response = chatbot.generate_response(prompt, results)
                
                # Display the intelligent response
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Display detailed results in expandable sections (original design)
                if results:
                    st.markdown("### 📚 Articles détaillés")
                    for i, r in enumerate(results):
                        with st.expander(f"{i+1}. {r.get('title', 'No Title')} (Sim: {r.get('similarity', 0):.3f})"):
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.markdown(f"**Auteurs**: {r.get('author', 'Unknown')}")
                            pub_date = r.get('published_date', 'Unknown')
                            if pd.notna(pub_date):
                                if isinstance(pub_date, str):
                                    try:
                                        pub_date = pd.to_datetime(pub_date).strftime('%Y-%m-%d')
                                    except:
                                        pass
                                else:
                                    try:
                                        pub_date = pub_date.strftime('%Y-%m-%d')
                                    except:
                                        pass
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
            error_msg = f"❌ Erreur lors de la recherche : {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.error(f"Erreur : {str(e)}")

    # --- TABS FOR ANALYTICS/ADVANCED SEARCH ---
    tab1, tab2 = st.tabs(["Analyses", "Recherche avancée"])
    with tab1:
        st.markdown("_Fonctionnalités d'analyse à venir..._")
    with tab2:
        st.markdown("_Recherche avancée à venir..._")

# Export the main function
__all__ = ['main']