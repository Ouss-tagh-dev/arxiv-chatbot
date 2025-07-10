"""
Enhanced Streamlit interface for the arXiv chatbot with:
- Optimized FAISS index loading
- Memory-efficient caching
- Improved search performance
- Better error handling
- Advanced filtering options
- Real-time monitoring
"""

import streamlit as st
from chatbot import ArxivChatbot
import time
import pandas as pd
import psutil
import gc
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_RESULTS = 50
CACHE_DIR = Path("data/cache")
EMBEDDING_DIR = Path("data/embeddings")

# Configure page
st.set_page_config(
    page_title="arXiv Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for optimized UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.75rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 15%;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #757575;
        margin-right: 15%;
    }
    .system-message {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 25%;
        font-size: 0.9rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .database-info {
        background-color: #f8f9fa;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stSpinner > div > div {
        border-top-color: #1f77b4;
    }
    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
    }
    .small-text {
        font-size: 0.85rem;
        color: #666;
    }
    .truncate {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    @media (max-width: 768px) {
        .chat-message {
            margin-left: 5% !important;
            margin-right: 5% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def get_system_info() -> Dict:
    """Get system resource information."""
    mem = psutil.virtual_memory()
    return {
        "memory_used": f"{mem.used / (1024**3):.1f} GB",
        "memory_total": f"{mem.total / (1024**3):.1f} GB",
        "memory_percent": f"{mem.percent}%",
        "cpu_usage": f"{psutil.cpu_percent()}%"
    }

def display_system_status():
    """Display current system status."""
    info = get_system_info()
    st.sidebar.markdown(f"""
    <div class='database-info'>
    <strong>🖥️ System Resources:</strong><br>
    <span class='small-text'>
    - Memory: {info['memory_used']} / {info['memory_total']} ({info['memory_percent']})<br>
    - CPU: {info['cpu_usage']}<br>
    - Threads: {psutil.Process().num_threads()}
    </span>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource(ttl=None, show_spinner="Initializing arXiv Research Assistant...")
def initialize_chatbot() -> ArxivChatbot:
    """Initialize and cache the chatbot with optimized loading."""
    start_time = time.time()
    
    try:
        # Create chatbot instance with optimized settings for large datasets
        chatbot = ArxivChatbot(
            max_memory_gb=8,  # Increased memory limit
            cache_dir=str(CACHE_DIR),
            enable_monitoring=True
        )
        
        # Verify index loading
        if not chatbot.search_engine or not hasattr(chatbot.search_engine, 'index'):
            raise RuntimeError("FAISS index not loaded properly")
        
        load_time = time.time() - start_time
        logger.info(f"Chatbot initialized in {load_time:.2f} seconds")
        
        # Store initialization metrics
        st.session_state.update({
            "chatbot_ready": True,
            "load_time": load_time,
            "total_articles": len(chatbot.df) if hasattr(chatbot, 'df') else 0,
            "model_name": chatbot.model_name
        })
        
        return chatbot
    
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise RuntimeError(f"Failed to initialize chatbot: {str(e)}")

def display_article_card(article: Dict, index: int) -> None:
    """Render an article card with optimized layout."""
    with st.expander(f"{index}. {article.get('title', 'No Title')[:120]}...", expanded=False):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Header with similarity score
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{article.get('title', 'No Title')}**")
        with col2:
            st.markdown(f"`Similarity: {article.get('similarity', 0):.3f}`")
        
        # Metadata row
        meta_col1, meta_col2, meta_col3 = st.columns([2, 2, 1])
        
        with meta_col1:
            authors = article.get('author', 'Unknown')
            if len(authors) > 60:
                authors = authors[:57] + "..."
            st.markdown(f"**Authors**: {authors}")
        
        with meta_col2:
            pub_date = article.get('published_date', 'Unknown')
            if isinstance(pub_date, str):
                try:
                    pub_date = pd.to_datetime(pub_date).strftime('%Y-%m-%d')
                except:
                    pass
            st.markdown(f"**Published**: {pub_date}")
        
        with meta_col3:
            cat = article.get('primary_category', article.get('category', ''))
            st.markdown(f"**Category**: {cat[:20]}" if cat else "**Category**: N/A")
        
        # Summary with read more/less functionality
        summary = article.get('summary', 'No summary available')
        if len(summary) > 300:
            if st.checkbox("Show full summary", key=f"summary_{index}"):
                st.markdown(f"**Summary**: {summary}")
            else:
                st.markdown(f"**Summary**: {summary[:300]}...")
        else:
            st.markdown(f"**Summary**: {summary}")
        
        # Links
        link_col1, link_col2 = st.columns(2)
        with link_col1:
            arxiv_id = article.get('id', '')
            if arxiv_id:
                st.markdown(f"[📄 arXiv Paper](https://arxiv.org/abs/{arxiv_id})")
        
        with link_col2:
            doi = article.get('doi', '')
            if doi and pd.notna(doi):
                st.markdown(f"[🌐 DOI](https://doi.org/{doi})")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_chat_message(role: str, content: str) -> None:
    """Render a chat message with appropriate styling."""
    with st.chat_message(role):
        if role == "assistant" and content.startswith("SYSTEM:"):
            st.markdown(f'<div class="system-message">{content[7:]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(content)

def main():
    """Main application interface."""
    st.markdown('<h1 class="main-header">arXiv Research Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "query_stats" not in st.session_state:
        st.session_state.query_stats = {}
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("## 🔍 Search Parameters")
        
        # Results control
        num_results = st.slider(
            "Number of results",
            min_value=1,
            max_value=MAX_RESULTS,
            value=10,
            help="Maximum number of articles to return"
        )
        
        # Filters
        st.markdown("### Filters")
        author_filter = st.text_input(
            "Author name contains",
            placeholder="e.g., Smith, Bengio, etc.",
            help="Filter by author name (partial match)"
        )
        
        # Initialize chatbot
        try:
            chatbot = initialize_chatbot()
            total_articles = len(chatbot.df) if hasattr(chatbot, 'df') else 0
            
            # Year filter
            if hasattr(chatbot, 'df') and 'published_date' in chatbot.df.columns:
                try:
                    chatbot.df['published_date'] = pd.to_datetime(chatbot.df['published_date'], errors='coerce')
                    years = sorted(chatbot.df['published_date'].dt.year.dropna().unique(), reverse=True)
                    year_filter = st.selectbox(
                        "Publication year",
                        ["All years"] + [str(int(y)) for y in years],
                        index=0
                    )
                except:
                    year_filter = "All years"
            else:
                year_filter = "All years"
            
            # Category filter
            if hasattr(chatbot, 'df') and 'primary_category' in chatbot.df.columns:
                categories = sorted([
                    c for c in chatbot.df["primary_category"].dropna().unique() 
                    if c and c != 'Unknown'
                ])
                selected_categories = st.multiselect(
                    "Categories",
                    categories,
                    default=categories[:3] if len(categories) >= 3 else categories,
                    help="Select one or more categories"
                )
            else:
                selected_categories = []
            
            # Advanced options
            with st.expander("Advanced Options"):
                search_fields = st.radio(
                    "Search fields",
                    ["All", "Titles only", "Content only"],
                    index=0,
                    horizontal=True
                )
                
                similarity_threshold = st.slider(
                    "Minimum similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="Filter out low similarity results"
                )
                
                rerank_method = st.selectbox(
                    "Rerank method",
                    ["Relevance", "Diversity", "Recent first"],
                    index=0
                )
            
            # Display database info
            st.markdown(f"""
            <div class='database-info'>
            <strong>📚 arXiv Database:</strong><br>
            <span class='small-text'>
            - Articles: {total_articles:,}<br>
            - Model: {st.session_state.get('model_name', 'N/A')}<br>
            - Load time: {st.session_state.get('load_time', 0):.2f}s
            </span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()
        
        display_system_status()
    
    # --- MAIN INTERFACE ---
    tab_chat, tab_results, tab_analytics = st.tabs(["Chat", "Results", "Analytics"])
    
    with tab_chat:
        # Display chat history
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                render_chat_message(message["role"], message["content"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about arXiv research..."):
            if not st.session_state.get("chatbot_ready", False):
                st.error("Chatbot is still initializing, please wait...")
                st.stop()
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            render_chat_message("user", prompt)
            
            try:
                # Prepare enhanced query with filters
                enhanced_query = prompt
                
                # Apply filters
                if year_filter != "All years":
                    enhanced_query += f" year:{year_filter}"
                if author_filter.strip():
                    enhanced_query += f" author:{author_filter.strip()}"
                if selected_categories:
                    enhanced_query += " " + " ".join(f"category:{cat}" for cat in selected_categories)
                
                # Perform search with progress bar
                with st.spinner("Searching arXiv database..."):
                    search_progress = st.progress(0)
                    
                    # Track query metrics
                    start_time = time.time()
                    results = chatbot.search_articles(
                        enhanced_query, 
                        k=num_results * 3  # Get more results for filtering
                    )
                    search_time = time.time() - start_time
                    
                    # Apply post-search filters
                    filtered_results = [
                        r for r in results 
                        if r.get('similarity', 0) >= similarity_threshold
                    ]
                    
                    # Apply reranking
                    if rerank_method == "Recent first" and 'published_date' in filtered_results[0]:
                        filtered_results.sort(
                            key=lambda x: x.get('published_date', ''),
                            reverse=True
                        )
                    elif rerank_method == "Diversity":
                        # Simple diversity - take top from each category
                        categories = {}
                        for res in filtered_results:
                            cat = res.get('primary_category', 'other')
                            if cat not in categories or res['similarity'] > categories[cat]['similarity']:
                                categories[cat] = res
                        filtered_results = list(categories.values())
                    
                    # Limit to requested number of results
                    filtered_results = filtered_results[:num_results]
                    
                    search_progress.progress(100)
                
                # Generate response
                with st.spinner("Generating response..."):
                    response = chatbot.generate_response(prompt, filtered_results)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    render_chat_message("assistant", response)
                
                # Store results and stats
                st.session_state.search_results = filtered_results
                st.session_state.query_stats = {
                    "query": prompt,
                    "time": search_time,
                    "total_results": len(results),
                    "filtered_results": len(filtered_results),
                    "similarity_threshold": similarity_threshold
                }
                
            except Exception as e:
                error_msg = f"SYSTEM: Error processing query: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                render_chat_message("assistant", error_msg)
                logger.error(f"Search error: {str(e)}")
    
    with tab_results:
        if st.session_state.search_results:
            st.markdown(f"### 📊 Found {len(st.session_state.search_results)} results")
            
            # Display stats
            if st.session_state.query_stats:
                stats = st.session_state.query_stats
                st.markdown(f"""
                <div class='small-text'>
                Query: "{stats['query']}"<br>
                Search time: {stats['time']:.2f}s | 
                Initial results: {stats['total_results']} | 
                After filtering: {stats['filtered_results']}
                </div>
                """, unsafe_allow_html=True)
            
            # Display results
            for i, result in enumerate(st.session_state.search_results, 1):
                display_article_card(result, i)
        else:
            st.info("No results to display. Perform a search in the Chat tab.")
    
    with tab_analytics:
        st.markdown("### 🔍 Search Analytics")
        
        if hasattr(chatbot, 'conversation_manager') and chatbot.conversation_manager.history:
            st.markdown("#### Conversation History")
            history_df = pd.DataFrame(chatbot.conversation_manager.history)
            st.dataframe(history_df[['timestamp', 'user', 'assistant']], height=300)
        
        if hasattr(chatbot, 'df') and not chatbot.df.empty:
            st.markdown("#### Category Distribution")
            category_counts = chatbot.df['primary_category'].value_counts().head(10)
            st.bar_chart(category_counts)
            
            if 'published_date' in chatbot.df.columns:
                st.markdown("#### Publication Trends")
                yearly_counts = chatbot.df['published_date'].dt.year.value_counts().sort_index()
                st.line_chart(yearly_counts)
        else:
            st.info("No analytics data available")

if __name__ == "__main__":
    main()