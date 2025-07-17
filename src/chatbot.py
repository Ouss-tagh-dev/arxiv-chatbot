"""
Enhanced conversational interface for the arXiv chatbot.
Optimized for large datasets with memory management and batch processing.
"""
from src.search_engine import OptimizedArxivSearchEngine as ArxivSearchEngine
from src.embedder import ArxivEmbedder
from src.data_loader import ArxivDataLoader
import openai
import os
from typing import List, Dict, Optional, Generator
import logging
import re
import json
import pickle
from pathlib import Path
import gc
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import io
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_RESULTS = 50
CACHE_DIR = Path("data/cache")
EMBEDDING_DIR = Path("data/embeddings")

class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.monitoring = False
        self.cleanup_callbacks = []
        
    def add_cleanup_callback(self, callback):
        """Add a callback to be called when memory cleanup is needed."""
        self.cleanup_callbacks.append(callback)
        
    def start_monitoring(self):
        """Start memory monitoring in a separate thread."""
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            memory_used = psutil.Process().memory_info().rss
            if memory_used > self.max_memory_bytes:
                logger.warning(f"Memory usage high: {memory_used / (1024**3):.2f} GB")
                self._trigger_cleanup()
            time.sleep(30)  # Check every 30 seconds
            
    def _trigger_cleanup(self):
        """Trigger cleanup callbacks."""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
        gc.collect()



class ConversationManager:
    """Manage conversation context with memory-efficient storage."""
    
    def __init__(self, max_history: int = 50, context_window: int = 6):
        self.max_history = max_history
        self.context_window = context_window
        self.history = []
        self.summary_cache = {}
        
    def add_exchange(self, user_message: str, assistant_response: str):
        """Add a conversation exchange."""
        self.history.append({
            'timestamp': time.time(),
            'user': user_message,
            'assistant': assistant_response
        })
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
    def get_context(self) -> List[Dict]:
        """Get recent conversation context for LLM."""
        recent_history = self.history[-self.context_window:]
        messages = []
        
        for exchange in recent_history:
            messages.append({"role": "user", "content": exchange['user']})
            messages.append({"role": "assistant", "content": exchange['assistant']})
            
        return messages
        
    def clear_old_context(self):
        """Clear old conversation context to free memory."""
        if len(self.history) > self.max_history // 2:
            self.history = self.history[-self.max_history // 2:]
            
    def save_to_disk(self, filepath: str):
        """Save conversation history to disk."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f)
            
    def load_from_disk(self, filepath: str):
        """Load conversation history from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.history = json.load(f)



class ArxivChatbot:
    def __init__(self, 
                 openai_api_key: str = None, 
                 model_name: str = "all-MiniLM-L6-v2",
                # model_name: str = "allenai/specter",
                 max_memory_gb: float = 4.0,
                 cache_dir: str = "data/cache",
                 enable_monitoring: bool = True):
        """
        Enhanced initialization optimized for large datasets.
        
        Args:
            openai_api_key: OpenAI API key for enhanced responses
            model_name: Sentence transformer model name
            max_memory_gb: Maximum memory usage in GB
            cache_dir: Directory for caching
            enable_monitoring: Whether to enable memory monitoring
        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.use_openai = bool(openai_api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(max_memory_gb)
        if enable_monitoring:
            self.memory_monitor.add_cleanup_callback(self._cleanup_memory)
            self.memory_monitor.start_monitoring()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
        # Initialize components lazily
        self.embedder = None
        self.search_engine = None
        self.loader = None
        self.df = None
        
        # Response cache
        self.response_cache = {}
        self.max_cache_size = 1000
        
        logger.info("ArxivChatbot initialized successfully")
        
    def _cleanup_memory(self):
        """Clean up memory when usage is high."""
        logger.info("Performing memory cleanup...")
        
        # Clear response cache
        self.response_cache.clear()
        
        # Clear conversation context
        self.conversation_manager.clear_old_context()
        
        # Clear embedder cache if needed
        if hasattr(self, 'embedder') and self.embedder:
            self.embedder.clear_cache()
            
        # Force garbage collection
        gc.collect()
        
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
    
    def cleanup(self):
        """Clean up resources when shutting down."""
        logger.info("Cleaning up chatbot resources...")
        self.memory_monitor.stop_monitoring()
        if hasattr(self, 'embedder') and self.embedder:
            self.embedder.clear_cache()
        gc.collect()

    def chat_loop(self):
        """Run an interactive chat loop in the terminal."""
        print("\nWelcome to the arXiv Research Assistant!")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                query = input("You: ")
                if query.lower() in ('quit', 'exit'):
                    break
                    
                response = self.generate_response(query)
                print("\nAssistant:", response, "\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


    def _initialize_components(self):
        """Initialize components only when needed."""
        if not self.embedder:
            logger.info("Initializing embedder...")
            self.embedder = ArxivEmbedder(model_name=self.model_name)
            
        if not self.search_engine:
            logger.info("Initializing search engine...")
            self.search_engine = ArxivSearchEngine()
            
            # Try to load pre-built index
            
            index_path = "data/embeddings/faiss_index_all-MiniLM-L6-v2.index"
            metadata_path = "data/embeddings/faiss_metadata_all-MiniLM-L6-v2.pkl"
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                logger.info("Loading existing FAISS index...")
                self.search_engine.load_index(index_path, metadata_path)
            else:
                logger.warning("No FAISS index found. Please run training first.")
                
        if not self.loader:
            logger.info("Initializing data loader...")
            self.loader = ArxivDataLoader("data/processed/articles_clean.csv")
            try:
                self.df = self.loader.load_data()
                if self.df is not None and not self.df.empty:
                    self.search_engine.load_article_data(self.df)
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                self.df = pd.DataFrame()


    def web_interface(self):
            """Run the chatbot with a Streamlit web interface."""

            self._is_web_interface = True
            self._initialize_components()
            # Configure page
            st.set_page_config(
                page_title="arXiv Research Assistant",
                page_icon="📚",
                layout="wide",
                initial_sidebar_state="expanded"
            )

            # Add custom CSS
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
            </style>
            """, unsafe_allow_html=True)

            # Initialize session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "search_results" not in st.session_state:
                st.session_state.search_results = []
            if "query_stats" not in st.session_state:
                st.session_state.query_stats = {}

            # Main header
            st.markdown('<h1 class="main-header">arXiv Research Assistant</h1>', unsafe_allow_html=True)

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
                
                # Year filter
                if hasattr(self, 'df') and 'published_date' in self.df.columns:
                    try:
                        self.df['published_date'] = pd.to_datetime(self.df['published_date'], errors='coerce')
                        years = sorted(self.df['published_date'].dt.year.dropna().unique(), reverse=True)
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
                if hasattr(self, 'df') and 'primary_category' in self.df.columns:
                    categories = sorted([
                        c for c in self.df["primary_category"].dropna().unique() 
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
                
                # Display system info
                mem = psutil.virtual_memory()
                st.markdown(f"""
                <div class='database-info'>
                <strong>🖥️ System Resources:</strong><br>
                Memory: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)<br>
                CPU: {psutil.cpu_percent()}%<br>
                Threads: {psutil.Process().num_threads()}
                </div>
                """, unsafe_allow_html=True)

            # --- MAIN INTERFACE ---
            tab_chat, tab_results = st.tabs(["Chat", "Results"])

            with tab_chat:
                # Display chat history
                with st.container():
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Chat input
                if prompt := st.chat_input("Ask about arXiv research..."):
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
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
                            results = self.search_articles(
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
                            response = self.generate_response(prompt, filtered_results)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
                        
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
                        error_msg = f"Error processing query: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        with st.chat_message("assistant"):
                            st.markdown(error_msg)
                        logger.error(f"Search error: {str(e)}")

            with tab_results:
                if st.session_state.search_results:
                    st.markdown(f"### 📊 Found {len(st.session_state.search_results)} results")
                    
                    # Display stats
                    if st.session_state.query_stats:
                        stats = st.session_state.query_stats
                        st.markdown(f"""
                        Query: "{stats['query']}"<br>
                        Search time: {stats['time']:.2f}s | 
                        Initial results: {stats['total_results']} | 
                        After filtering: {stats['filtered_results']}
                        """, unsafe_allow_html=True)
                    
                    # Display results
                    for i, result in enumerate(st.session_state.search_results, 1):
                        with st.expander(f"{i}. {result.get('title', 'No Title')[:120]}...", expanded=False):
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            
                            # Header with similarity score
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{result.get('title', 'No Title')}**")
                            with col2:
                                st.markdown(f"`Similarity: {result.get('similarity', 0):.3f}`")
                            
                            # Metadata row
                            meta_col1, meta_col2, meta_col3 = st.columns([2, 2, 1])
                            
                            with meta_col1:
                                authors = result.get('author', 'Unknown')
                                if len(authors) > 60:
                                    authors = authors[:57] + "..."
                                st.markdown(f"**Authors**: {authors}")
                            
                            with meta_col2:
                                pub_date = result.get('published_date', 'Unknown')
                                if isinstance(pub_date, str):
                                    try:
                                        pub_date = pd.to_datetime(pub_date).strftime('%Y-%m-%d')
                                    except:
                                        pass
                                st.markdown(f"**Published**: {pub_date}")
                            
                            with meta_col3:
                                cat = result.get('primary_category', result.get('category', ''))
                                st.markdown(f"**Category**: {cat[:20]}" if cat else "**Category**: N/A")
                            
                            # Summary
                            summary = result.get('summary', 'No summary available')
                            st.markdown(f"**Summary**: {summary}")
                            
                            # Links
                            link_col1, link_col2 = st.columns(2)
                            with link_col1:
                                arxiv_id = result.get('id', '')
                                if arxiv_id:
                                    st.markdown(f"[📄 arXiv Paper](https://arxiv.org/abs/{arxiv_id})")
                            
                            with link_col2:
                                doi = result.get('doi', '')
                                if doi and pd.notna(doi):
                                    st.markdown(f"[🌐 DOI](https://doi.org/{doi})")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No results to display. Perform a search in the Chat tab.")


    def detect_intent(self, query: str) -> str:
        """
        Enhanced intent detection that works for any arXiv topic.
        Returns the most likely intent based on query patterns.
        """
        q = query.lower().strip()
        
        # Statistical queries - look for counting/quantity words
        stat_patterns = [
            (r"(how many|number of|count of|total) (papers|articles|studies|publications)", "statistics"),
            (r"(statistics|stats|analysis) (of|on|for|about)", "statistics"),
            (r"distribution of (papers|articles|studies)", "statistics"),
            (r"percentage of (papers|articles) (in|about)", "statistics"),
            (r"trends? in (publications|papers|research)", "statistics"),
            (r"most (active|productive|published) authors?", "statistics"),
            (r"popular (categories|fields|topics)", "statistics")
        ]
        
        for pattern, intent in stat_patterns:
            if re.search(pattern, q):
                return intent

        
        # Author queries - look for people/researcher focused terms
        author_keywords = [
            "author", "written by", "who wrote", "papers by", "work by",
            "researcher", "scientist", "published by", "research by",
            "by ", " by", "from "  # Common author patterns
        ]
        if any(phrase in q for phrase in author_keywords):
            return "author"
        
        # Recent/temporal queries
        recent_keywords = [
            "recent", "latest", "newest", "new", "current", "today",
            "2023", "2024", "2025", "this year", "last year", "modern",
            "up to date", "fresh", "contemporary"
        ]
        if any(phrase in q for phrase in recent_keywords):
            return "recent"
        
        # Summary queries
        summary_keywords = [
            "summarize", "summary", "explain", "tell me about", "overview",
            "briefly", "in short", "key points", "main ideas", "describe",
            "what does", "give me an overview"
        ]
        if any(phrase in q for phrase in summary_keywords):
            return "summary"
        
        # Comparison queries
        comparison_keywords = [
            "compare", "contrast", "difference between", "vs", "versus",
            "better than", "differ", "distinction", "similar to",
            "relationship between", "against", "comparison of"
        ]
        if any(phrase in q for phrase in comparison_keywords):
            return "comparison"
        
        # Trend queries
        trend_keywords = [
            "trend", "evolution", "over time", "history of", "progress",
            "development", "advancement", "timeline", "growth", "decline",
            "patterns", "changes", "evolution of", 
            "trending categories", "which categories are trending", "hottest fields"
        ]

        if any(phrase in q for phrase in trend_keywords):
            return "trend"
        
        # Definition queries
        definition_keywords = [
            "what is", "define", "definition", "meaning", "concept of",
            "explain", "understand", "about", "introduction to"
        ]
        if any(phrase in q for phrase in definition_keywords):
            return "definition"
        
        # Recommendation queries
        recommendation_keywords = [
            "recommend", "suggestion", "best", "top", "good papers",
            "should i", "which one", "important", "must read",
            "essential", "suggest", "recommendations"
        ]
        if any(phrase in q for phrase in recommendation_keywords):
            return "recommendation"
        
        # Category intent detection
        category_keywords = [
            "category", "show stats for", "summary of category", 
            "papers in category", "trending in", "what's new in"
        ]
        if any(kw in q for kw in category_keywords):
            return "category"

        # Default to search for everything else
        return "search"




    def search_articles(self, query: str, k: int = 10) -> List[Dict]:
        """
        Search for articles with caching and error handling.
        """
        self._initialize_components()
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.response_cache:
            logger.info(f"Returning cached results for query: {query[:50]}...")
            return self.response_cache[cache_key]
            
        try:
            # Perform search
            results = self.search_engine.search_by_text(query, self.embedder, k, search_fields="all")
            
            # Cache results
            if len(self.response_cache) >= self.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.response_cache.keys())[:100]
                for key in oldest_keys:
                    del self.response_cache[key]
                    
            self.response_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []
            
    def generate_response(self, query: str, search_results: List[Dict] = None) -> str:
        """
        Enhanced response generation with conversation context and caching.
        """
        if search_results is None:
            search_results = self.search_articles(query)
            
        intent = self.detect_intent(query)
        
        if not search_results:
            response = self._generate_no_results_response(query, intent)
        else:
            # Generate context for LLM
            context = self._build_response_context(query, search_results, intent)
            
            # Use OpenAI if available for high-quality responses
            if self.use_openai:
                try:
                    response = self._generate_llm_response(query, context, intent)
                except Exception as e:
                    logger.error(f"OpenAI error: {e}")
                    response = self._generate_local_response(query, search_results, intent)
            else:
                response = self._generate_local_response(query, search_results, intent)
        
        # Add to conversation history
        self.conversation_manager.add_exchange(query, response)
        
        return response
        
    def _generate_no_results_response(self, query: str, intent: str) -> str:
        """Generate response when no results are found."""
        suggestions = [
            "Try using more general terms",
            "Check for spelling errors",
            "Use different keywords",
            "Try searching in English if you used another language"
        ]
        
        return f"I couldn't find any relevant articles for '{query}'. " \
               f"Here are some suggestions:\n" + "\n".join(f"• {s}" for s in suggestions)
        
    def _build_response_context(self, query: str, results: List[Dict], intent: str) -> str:
        """
        Build context string for LLM response generation.
        """
        context = f"User query: {query}\n"
        context += f"Intent detected: {intent}\n"
        context += f"Found {len(results)} relevant articles:\n\n"
        
        # Include conversation context
        recent_context = self.conversation_manager.get_context()
        if recent_context:
            context += "Recent conversation:\n"
            for msg in recent_context[-4:]:  # Last 2 exchanges
                context += f"{msg['role']}: {msg['content'][:100]}...\n"
            context += "\n"
        
        # Add article details
        for i, res in enumerate(results[:3], 1):  # Limit to top 3 for context
            context += f"Article {i}:\n"
            context += f"Title: {res.get('title', 'N/A')}\n"
            context += f"Authors: {res.get('author', 'N/A')[:200]}...\n"
            context += f"Published: {res.get('published_date', 'N/A')}\n"
            context += f"Categories: {res.get('primary_category', 'N/A')}\n"
            
            summary = res.get('summary', '')
            if len(summary) > 300:
                summary = summary[:300] + "..."
            context += f"Summary: {summary}\n"
            context += f"Similarity: {res.get('similarity', 0):.3f}\n\n"
            
        return context
        


    def _generate_llm_response(self, query: str, context: str, intent: str) -> str:
        """Generate response using OpenAI's API with conversation context."""
        system_prompt = """You are a knowledgeable and friendly scientific assistant specializing in arXiv research papers. 
        Your responses should be:
        - Natural and conversational, like a helpful expert
        - Detailed but concise
        - Well-formatted with clear structure
        - Include relevant paper details when appropriate
        - Maintain context from the conversation
        
        When discussing papers:
        1. Start with a general response to the query
        2. Highlight key findings or relevance
        3. Provide specific details from the most relevant papers
        4. Offer additional insights or connections
        
        Use markdown for formatting with **bold**, *italics*, and lists when helpful."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_manager.get_context(),
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
        ]
        
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-4" if "gpt-4" in str(self.openai_api_key) else "gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                timeout=30
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise    
    


    def _generate_local_response(self, query: str, results: List[Dict], intent: str) -> str:
        """
        Generate natural responses without LLM using enhanced templates.
        Fixed to properly route to the correct response generator.
        """
        if not results:
            return self._generate_no_results_response(query, intent)
        
        # Enhanced response generators with proper routing
        response_generators = {
            "statistics": self._generate_statistics_response,
            "author": self._generate_author_response,
            "recent": self._generate_recent_response,
            "summary": self._generate_summary_response,
            "comparison": self._generate_comparison_response,
            "trend": self._generate_trend_response,
            "definition": self._generate_definition_response,
            "recommendation": self._generate_recommendation_response,
            "category": self._generate_category_response,
            "search": self._generate_search_response  # Default fallback
        }
        
        # Get the appropriate generator function
        generator = response_generators.get(intent, self._generate_search_response)
        
        try:
            return generator(query, results)
        except Exception as e:
            logger.error(f"Error in response generation for intent '{intent}': {e}")
            # Fallback to search response
            return self._generate_search_response(query, results)


    def _generate_search_response(self, query: str, results: List[Dict]) -> str:
        """Generate a comprehensive search response for any arXiv topic."""
        if not results:
            return self._generate_no_results_response(query, "search")
        
        top_result = results[0]
        
        response = f"**Search results for '{query}':**\n\n"
        response += f"Found {len(results)} relevant papers. Here are the most relevant ones:\n\n"
        
        # Main result with enhanced formatting
        response += f"**🏆 Top Result:**\n"
        response += f"**{top_result.get('title', 'Untitled')}**\n"
        response += f"👥 *{top_result.get('author', 'Unknown authors')}*\n"
        
        # Handle publication date (could be string, Timestamp, or other format)
        pub_date = top_result.get('published_date', 'Unknown date')
        if hasattr(pub_date, 'strftime'):  # If it's a Timestamp/datetime object
            pub_date = pub_date.strftime('%Y-%m-%d')
        elif isinstance(pub_date, str) and len(pub_date) > 10:  # If it's a long string
            pub_date = pub_date[:10]
        response += f"📅 *{pub_date}*\n"
        
        response += f"📂 *{top_result.get('primary_category', 'Unknown category')}*\n"
        response += f"🎯 *Relevance: {top_result.get('similarity', 0):.1%}*\n\n"
        
        # Summary with smart truncation
        summary = top_result.get('summary', 'No summary available')
        if summary and summary != 'No summary available':
            if len(summary) > 400:
                sentences = summary.split('. ')
                if len(sentences) > 3:
                    summary = '. '.join(sentences[:3]) + '...'
                else:
                    summary = summary[:400] + "..."
            response += f"**📝 Abstract:** {summary}\n\n"
        
        # Additional results with compact formatting
        if len(results) > 1:
            response += f"**📚 Other relevant papers:**\n"
            for i, paper in enumerate(results[1:5], 2):  # Show up to 4 more
                title = paper.get('title', 'Untitled')
                if len(title) > 70:
                    title = title[:70] + "..."
                
                author = paper.get('author', 'Unknown')
                if len(author) > 50:
                    author = author[:50] + "..."
                
                # Handle publication date for additional papers
                paper_date = paper.get('published_date', 'Unknown')
                if hasattr(paper_date, 'strftime'):
                    paper_date = paper_date.strftime('%Y-%m-%d')
                elif isinstance(paper_date, str) and len(paper_date) > 10:
                    paper_date = paper_date[:10]
                
                response += f"**{i}.** {title}\n"
                response += f"   👥 {author} | "
                response += f"📅 {paper_date} | "
                response += f"📂 {paper.get('primary_category', 'Unknown')}\n"
                response += f"   🎯 {paper.get('similarity', 0):.1%}\n\n"
        
        # Quick analytics
        response += f"**📊 Quick Analysis:**\n"
        
        # Category distribution
        categories = {}
        for res in results:
            cat = res.get('primary_category', 'Unknown')
            if cat != 'Unknown':
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            response += f"• **Main research areas:** {', '.join([cat for cat, _ in top_cats])}\n"
        
        # Year range
        years = []
        for res in results:
            date = res.get('published_date', '')
            if hasattr(date, 'year'):  # If it's a Timestamp/datetime object
                years.append(str(date.year))
            elif isinstance(date, str) and len(date) >= 4:  # If it's a string with year
                years.append(date[:4])
        
        if years:
            response += f"• **Publication range:** {min(years)} - {max(years)}\n"
        
        # Suggestion for follow-up
        response += f"\n💡 **Want to explore more?** Try:\n"
        response += f"• 'Recent papers on {query}' for latest research\n"
        response += f"• 'Statistics on {query}' for detailed analysis\n"
        response += f"• 'Summarize {query}' for key insights\n"
        
        return response
    



    def _generate_statistics_response(self, query: str, results: List[Dict]) -> str:
        """Generate comprehensive statistics with visualizations."""
        total = len(results)
        
        if total == 0:
            return self._generate_no_results_response(query, "statistics")
        
        # Enhanced analysis
        categories = {}
        years = {}
        authors = {}
        unique_authors = set()
        
        for res in results:
            # Categories
            cat = res.get('primary_category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Years - enhanced parsing
            year = self._extract_year(res.get('published_date', ''))
            if year:
                years[year] = years.get(year, 0) + 1
            
            # Authors - enhanced parsing
            author_list = self._parse_authors(res.get('author', ''))
            for author in author_list:
                if author:
                    unique_authors.add(author)
                    authors[author] = authors.get(author, 0) + 1
        
        # Prepare response
        response = f"**📊 Comprehensive Statistics for '{query}':**\n\n"
        response += f"**🔢 Overview:**\n"
        response += f"• Total papers: **{total}**\n"
        response += f"• Unique authors: **{len(unique_authors)}**\n"
        response += f"• Publication years: **{len(years)}**\n"
        response += f"• Research categories: **{len(categories)}**\n\n"
        
        # In web interface, display charts directly
        if hasattr(self, '_is_web_interface') and self._is_web_interface:
            import streamlit as st
            
            if categories:
                fig = self._generate_category_chart(categories, query)
                st.pyplot(fig)
                plt.close(fig)
            
            if years and len(years) > 1:
                fig = self._generate_trend_chart(years, query)
                st.pyplot(fig)
                plt.close(fig)
            
            if authors:
                fig = self._generate_author_chart(authors, query)
                st.pyplot(fig)
                plt.close(fig)
        else:
            # For non-web interface, provide textual statistics
            if categories:
                response += "**📂 Top Research Categories:**\n"
                top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (cat, count) in enumerate(top_cats, 1):
                    percentage = (count / total) * 100
                    response += f"{i}. **{cat}**: {count} papers ({percentage:.1f}%)\n"
            
            if years:
                response += "\n**📅 Publication Timeline:**\n"
                sorted_years = sorted(years.items())
                for year, count in sorted_years[-5:]:  # Show most recent 5 years
                    response += f"• **{year}**: {count} papers\n"
            
            if authors:
                response += "\n**👥 Most Prolific Authors:**\n"
                top_auths = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (author, count) in enumerate(top_auths, 1):
                    response += f"{i}. **{author}**: {count} papers\n"
        
        # Trend analysis
        if len(years) > 2:
            response += "\n**📈 Trend Analysis:**\n"
            sorted_years = sorted(years.items())
            mid_point = len(sorted_years) // 2
            early = sum(count for _, count in sorted_years[:mid_point]) / mid_point
            recent = sum(count for _, count in sorted_years[mid_point:]) / (len(sorted_years) - mid_point)
            
            if recent > early * 1.2:
                response += f"• **Growing field** (+{((recent/early)-1)*100:.1f}% increase)\n"
            elif recent < early * 0.8:
                response += f"• **Declining activity** (-{(1-(recent/early))*100:.1f}% decrease)\n"
            else:
                response += "• **Stable publication rate**\n"
        
        # Add suggestions for further exploration
        response += "\n**💡 Explore Further:**\n"
        response += "• 'Show me papers from [year]' - Filter by specific year\n"
        response += "• 'Papers by [author]' - Find works by specific researchers\n"
        response += "• 'Trends in [category]' - Analyze specific research areas\n"
        
        return response

    def _extract_year(self, date) -> Optional[str]:
        """Extract year from various date formats."""
        if isinstance(date, pd.Timestamp):
            return str(date.year)
        if isinstance(date, str):
            if len(date) >= 4 and date[:4].isdigit():
                return date[:4]
        return None


    def _parse_authors(self, authors_str: str) -> List[str]:
        """Parse authors string into list of normalized names."""
        if not authors_str or pd.isna(authors_str):
            return []
        
        # Handle different formats
        if authors_str.startswith('[') and authors_str.endswith(']'):
            # Handle bracketed format like ["Author 1", "Author 2"]
            try:
                import ast
                return [a.strip() for a in ast.literal_eval(authors_str) if a.strip()]
            except:
                # Fallback to splitting if eval fails
                authors_str = authors_str[1:-1]  # Remove brackets
        
        # Common delimiters in author strings
        delimiters = [';', ',', ' and ']
        
        # Split by the first delimiter found
        for delim in delimiters:
            if delim in authors_str:
                return [a.strip().strip('"').strip("'") 
                    for a in authors_str.split(delim) 
                    if a.strip()]
        
        return [authors_str.strip()]
    

    def _generate_category_chart(self, categories: Dict[str, int], query: str):
        """Generate a category distribution chart and return figure."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
        cats, counts = zip(*sorted_cats)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(cats), palette="viridis", ax=ax)
        ax.set_title(f"Category Distribution for '{query}'")
        ax.set_xlabel("Number of Papers")
        ax.set_ylabel("Category")
        
        return fig

    def _generate_trend_chart(self, years: Dict[str, int], query: str):
        """Generate a publication trend chart."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data
        sorted_years = sorted(years.items())
        years_list, counts = zip(*sorted_years)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=list(years_list), y=list(counts), marker='o', ax=ax)
        ax.set_title(f"Publication Trend for '{query}'")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Papers")
        plt.xticks(rotation=45)
        plt.grid(True)
        
        return fig

    def _generate_author_chart(self, authors: Dict[str, int], query: str):
        """Generate a top authors chart."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data
        sorted_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10]
        authors_list, counts = zip(*sorted_authors)
        
        # Shorten author names
        authors_short = [name[:20] + '...' if len(name) > 20 else name for name in authors_list]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(counts), y=authors_short, palette="rocket", ax=ax)
        ax.set_title(f"Top Authors for '{query}'")
        ax.set_xlabel("Number of Papers")
        ax.set_ylabel("Author")
        
        return fig

    def _generate_author_response(self, query: str, results: List[Dict]) -> str:
        """
        Smartly decide whether the user is asking:
        (A) "Who wrote <paper title>?"  → return authors of that paper
        (B) "written by <author>"       → list up to 10 papers by that author

        `results` should be the already-available search results for the *raw* query
        (typically semantic search over title+abstract+authors). We'll reuse them
        for author-mode; in paper-mode we re-query by the extracted title.
        """
        import re

        q = (query or "").strip()
        if not q:
            return "I didn't catch the author or title. Please try again."

        # ------------------------------------------------------------
        # Patterns
        # ------------------------------------------------------------
        who_pattern = r"^(?:who\s+wrote|who\s+is\s+the\s+author\s+of|author\s+of)\s+(.+)$"
        author_pattern = r"^(?:written\s+by|papers?\s+by|by|author:)\s+(.+)$"

        who_match = re.search(who_pattern, q, re.IGNORECASE)
        author_match = re.search(author_pattern, q, re.IGNORECASE)

        # ------------------------------------------------------------
        # Ambiguity guard: If user wrote "written by X" but X looks like a paper title
        # (many tokens, stopwords), we'll flip to who-mode. Conversely, if user wrote
        # "author of X" but X looks like a short person name, flip to author-mode.
        # ------------------------------------------------------------
        def _looks_like_author_name(text: str) -> bool:
            return self._looks_like_author_name(text)

        # ------------------------------------------------------------
        # Branch 1: explicit *who wrote / author of* pattern → paper-mode
        # ------------------------------------------------------------
        if who_match:
            payload = who_match.group(1).strip()
            # If payload *actually* looks like a name, redirect to author-mode
            if _looks_like_author_name(payload):
                return self._respond_author_mode(payload, results)
            return self._respond_who_mode(payload)

        # ------------------------------------------------------------
        # Branch 2: explicit *written by / papers by / by / author:* → author-mode
        # ------------------------------------------------------------
        if author_match:
            payload = author_match.group(1).strip()
            # If payload seems like a long / title-like thing, flip to who-mode
            if not _looks_like_author_name(payload):
                return self._respond_who_mode(payload)
            return self._respond_author_mode(payload, results)

        # ------------------------------------------------------------
        # Branch 3: no explicit pattern → infer by heuristic
        # Long, title-like text → who-mode. Short, name-like → author-mode.
        # ------------------------------------------------------------
        if _looks_like_author_name(q):
            return self._respond_author_mode(q, results)
        else:
            return self._respond_who_mode(q)


    def _respond_author_mode(self, author_query: str, results: List[Dict]) -> str:
        author_pattern = author_query.strip().strip('"').strip("'")
        author_papers = []

        print(f"\nDEBUG: Searching for author '{author_query}'")

        # ✅ Prefer full dataset if available
        if hasattr(self, 'df') and isinstance(self.df, pd.DataFrame):
            for i, row in self.df.iterrows():
                authors = row.get('author', '')
                if self._author_match(author_pattern, authors):
                    paper = row.to_dict()
                    paper['similarity'] = 1.0  # Optional: treat as max relevance
                    author_papers.append(paper)
        else:
            # fallback to semantic results
            for i, paper in enumerate(results):
                authors = paper.get('author', '')
                if self._author_match(author_pattern, authors):
                    author_papers.append(paper)

        if not author_papers:
            return f"No papers found matching author '{author_pattern}'. Try adjusting spelling (e.g., initials vs full name)."

        # Sort newest first
        author_papers.sort(key=lambda x: x.get('published_date', ''), reverse=True)

        response = f"Papers by {author_pattern}:\n\n"
        response += f"Found {len(author_papers)} papers:\n\n"

        for i, paper in enumerate(author_papers[:10], 1):
            response += f"{i}. {paper.get('title', 'Untitled')}\n"
            response += f"   📅 Published: {self._format_pub_date(paper.get('published_date'))}\n"
            response += f"   👥 Authors: {paper.get('author', 'Unknown authors')}\n"
            response += f"   📂 Category: {paper.get('primary_category', 'Unknown category')}\n"
            response += f"   {self._format_links_block(paper, indent='   ')}\n\n"

        return response.rstrip()


    def _respond_who_mode(self, paper_title_query: str) -> str:
        """
        Search specifically for a paper title and report its author list + metadata.
        """
        paper_results = self.search_articles(paper_title_query, k=5)
        if not paper_results:
            return f"I couldn't find the paper titled '{paper_title_query}'. Please double-check the title."

        top_paper = paper_results[0]
        authors = top_paper.get('author', 'Unknown authors')

        response = f"The paper '{top_paper.get('title', paper_title_query)}' was written by:\n"
        response += f"👥 {authors}\n\n"
        response += f"📅 Published: {self._format_pub_date(top_paper.get('published_date'))}\n"
        response += f"📂 Category: {top_paper.get('primary_category', 'Unknown category')}\n"
        links_block = self._format_links_block(top_paper)
        if links_block:
            response += links_block
        return response


    def _format_pub_date(self, pub_date) -> str:
        """Return date string in 'YYYY-MM-DD HH:MM:SS+00:00' when possible."""
        if pub_date in (None, '', 'Unknown date'):
            return 'Unknown date'
        try:
            if hasattr(pub_date, 'strftime'):
                return pub_date.strftime('%Y-%m-%d %H:%M:%S+00:00')
            pub_date = str(pub_date)
            # if it's long but includes date at start
            if len(pub_date) >= 10:
                return pub_date[:10]
            return pub_date
        except Exception:
            return 'Unknown date'

    def _format_links_block(self, paper: Dict, indent: str = "") -> str:
        links = []
        arxiv_id = paper.get('id')
        if arxiv_id:
            # For single-line link group we prefer Abstract; PDF optional; Search fallback rarely needed
            links.append(f"🔍 [arXiv Abstract](https://arxiv.org/abs/{arxiv_id})")
            links.append(f"📄 [PDF](https://arxiv.org/pdf/{arxiv_id}.pdf)")
        doi = paper.get('doi')
        if doi and str(doi).strip().lower() not in ('nan', 'none', ''):
            links.append(f"🌐 [DOI](https://doi.org/{doi})")

        if not links:
            title = paper.get('title', '')
            if title:
                search_query = title.replace(' ', '+')[:100]
                links.append(f"🔎 [Search on arXiv](https://arxiv.org/search/?query={search_query}&searchtype=title)")

        if not links:
            return ""
        if indent:
            return indent + " | ".join(links)
        return "🔗 " + " | ".join(links)


    def _author_match(self, pattern: str, authors: str) -> bool:
        """Strict matching: match only if full name is present among parsed authors."""
        pattern_norm = self._normalize_name(pattern)

        author_list = self._parse_authors(authors)
        for author in author_list:
            author_norm = self._normalize_name(author)
            if pattern_norm == author_norm:
                return True
        return False


    def _normalize_name(self, name: str) -> str:
        import unicodedata
        import re
        if not name:
            return ""
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        name = re.sub(r'[^\w\s-]', '', name.lower())
        return name.strip()


    def _looks_like_author_name(self, text: str) -> bool:
        """Improved heuristic for detecting author names."""
        import re
        
        t = text.strip().strip('"').strip("'")
        if not t:
            return False
        
        # Common academic name patterns
        name_patterns = [
            r"^[A-Z][a-z]+ [A-Z][a-z]+$",  # First Last
            r"^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+$",  # First M. Last
            r"^[A-Z]\. [A-Z][a-z]+$",  # M. Last
            r"^[A-Z][a-z]+-[A-Z][a-z]+ [A-Z][a-z]+$",  # Hyphenated names
        ]
        
        if any(re.match(p, t) for p in name_patterns):
            return True
        
        tokens = re.split(r'\s+', t)
        
        # Quick length check
        if len(tokens) > 6:  # very long -> probably a title
            return False
        
        # Common title stopwords
        stop = {"for","of","in","on","with","and","the","a","an","to",
                "from","via","using","into","behavior","network",
                "errors","algorithm","study","analysis","model"}
        stop_hits = sum(1 for tok in tokens if tok.lower() in stop)
        if stop_hits >= 2:
            return False
        
        # If most tokens start uppercase letter OR are initials
        name_like = 0
        for tok in tokens:
            if re.match(r"^[A-Z][\w\-'.]*$", tok) or re.match(r"^[A-Z]\.?$", tok):
                name_like += 1
        if name_like >= max(1, len(tokens) - 1):
            return True
        
        # If contains a hyphenated cap token and token count <=4
        if any('-' in tok for tok in tokens) and len(tokens) <= 4:
            return True
        
        # Fallback: short (<=3 tokens) & no stopwords
        return len(tokens) <= 3 and stop_hits == 0


    def _generate_recent_response(self, query: str, results: List[Dict]) -> str:
        """Generate recent papers response for any topic."""
        if not results:
            return self._generate_no_results_response(query, "recent")
        
        # Sort by publication date (most recent first)
        sorted_results = sorted(results, 
                            key=lambda x: x.get('published_date', ''), 
                            reverse=True)
        
        # Filter for truly recent papers (last 2 years)
        current_year = 2025
        recent_papers = []
        
        for res in sorted_results:
            pub_date = res.get('published_date', '')
            if pub_date:
                year = pub_date.strftime('%Y') if isinstance(pub_date, pd.Timestamp) else pub_date[:4]
                if year.isdigit() and int(year) >= current_year - 2:
                    recent_papers.append(res)
        
        if not recent_papers:
            # If no recent papers, show the most recent ones available
            recent_papers = sorted_results[:5]
        
        response = f"**Recent papers on '{query}':**\n\n"
        
        if recent_papers == sorted_results[:5]:
            response += f"Found {len(recent_papers)} most recent papers (may not be from last 2 years):\n\n"
        else:
            response += f"Found {len(recent_papers)} papers from the last 2 years:\n\n"
        
        for i, paper in enumerate(recent_papers[:5], 1):
            response += f"**{i}. {paper.get('title', 'Untitled')}**\n"
            response += f"   📅 {paper.get('published_date', 'Unknown date')}\n"
            response += f"   👥 {paper.get('author', 'Unknown authors')[:80]}...\n"
            response += f"   📂 {paper.get('primary_category', 'Unknown category')}\n\n"
        
        return response 


    def _generate_summary_response(self, query: str, results: List[Dict]) -> str:
        """Generate a comprehensive summary response for any topic."""
        if not results:
            return self._generate_no_results_response(query, "summary")
        
        top_result = results[0]
        
        response = f"**Summary of research on '{query}':**\n\n"
        
        # Main paper summary
        response += f"**Key Paper: {top_result.get('title', 'Untitled')}**\n"
        response += f"*Authors: {top_result.get('author', 'Unknown')}*\n"
        response += f"*Published: {top_result.get('published_date', 'Unknown')}*\n\n"
        
        # Process summary
        summary = top_result.get('summary', 'No summary available')
        if len(summary) > 500:
            sentences = summary.split('. ')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]) + '...'
            else:
                summary = summary[:500] + "..."
        
        response += f"**Research Overview:** {summary}\n\n"
        
        # Additional context from other papers
        if len(results) > 1:
            response += f"**Related Research:**\n"
            response += f"Found {len(results)} total papers on this topic. "
            
            # Analyze categories
            categories = {}
            for res in results:
                cat = res.get('primary_category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            response += f"Main research areas: {', '.join([cat for cat, _ in top_cats])}\n\n"
        
        response += "💡 **Would you like me to:**\n"
        response += "• Provide more details about specific papers\n"
        response += "• Show recent developments in this area\n"
        response += "• Find papers by specific authors\n"
        
        return response


    def _generate_trend_response(self, query: str, results: List[Dict]) -> str:
        """Generate trend analysis with year timeline and trending categories."""
        if not results:
            return self._generate_no_results_response(query, "trend")

        years = {}
        categories_by_year = {}

        for res in results:
            pub_date = res.get('published_date', '')
            if isinstance(pub_date, pd.Timestamp):
                year = pub_date.strftime('%Y')
            elif isinstance(pub_date, str):
                year = pub_date[:4]
            else:
                year = ''
            if year.isdigit():
                years[year] = years.get(year, 0) + 1
                cat = res.get('primary_category', 'Unknown')
                categories_by_year.setdefault(year, {})
                categories_by_year[year][cat] = categories_by_year[year].get(cat, 0) + 1

        if not years:
            return f"No publication dates found for papers on '{query}'."

        sorted_years = sorted(years.items(), key=lambda x: int(x[0]))
        year_lines = [f"• **{y}**: {c} paper{'s' if c != 1 else ''}" for y, c in sorted_years]
        year_block = '\n'.join(year_lines)

        response = f"**📈 Publication Trends for '{query.strip(' ?')}'**\n\n"
        response += f"**🗓️ Timeline Overview:**\n{year_block}\n\n"

        # Trend interpretation
        if len(sorted_years) > 2:
            recent_years = sorted_years[-3:]
            earlier_years = sorted_years[:-3]

            recent_avg = sum(c for _, c in recent_years) / len(recent_years)
            earlier_avg = sum(c for _, c in earlier_years) / len(earlier_years) if earlier_years else 0

            response += "**📊 Trend Analysis:**\n"
            if earlier_avg == 0 and recent_avg > 0:
                response += "• 🚀 **Emerging topic** — no earlier activity, but recent publications show interest\n"
            elif recent_avg > earlier_avg * 1.3:
                response += f"• 📈 **Rising interest** — recent avg: {recent_avg:.1f} vs earlier avg: {earlier_avg:.1f}\n"
            elif recent_avg < earlier_avg * 0.7:
                response += f"• 📉 **Declining interest** — recent avg: {recent_avg:.1f} vs earlier avg: {earlier_avg:.1f}\n"
            else:
                response += f"• 📊 **Stable activity** — recent avg: {recent_avg:.1f} vs earlier avg: {earlier_avg:.1f}\n"

        peak_year, peak_count = max(years.items(), key=lambda x: x[1])
        response += f"• 🏆 **Peak year:** {peak_year} with {peak_count} paper{'s' if peak_count != 1 else ''}\n"

        # Optional: highlight recent categories
        if categories_by_year:
            latest_year = sorted(categories_by_year.keys())[-1]
            recent_cats = categories_by_year[latest_year]
            top_cat = max(recent_cats.items(), key=lambda x: x[1])[0]
            response += f"• 🧠 **Most active category in {latest_year}**: {top_cat}\n"

        # 🔥 Trending categories
        response += "\n**🔥 Trending Research Categories:**\n"
        category_growth = {}

        for year in sorted(categories_by_year.keys()):
            for cat, count in categories_by_year[year].items():
                category_growth.setdefault(cat, [])
                category_growth[cat].append((year, count))

        growth_scores = []
        for cat, yearly_counts in category_growth.items():
            if len(yearly_counts) < 2:
                continue
            yearly_counts.sort()
            first_year, first_count = yearly_counts[0]
            last_year, last_count = yearly_counts[-1]
            growth = (last_count - first_count) / max(1, first_count)
            growth_scores.append((cat, growth, first_count, last_count))

        top_trending = sorted(growth_scores, key=lambda x: x[1], reverse=True)[:3]

        if top_trending:
            for cat, growth, start, end in top_trending:
                trend_type = "📈 Rising" if growth > 0 else "📉 Declining"
                response += f"• {trend_type} **{cat}** — from {start} to {end} papers\n"
        else:
            response += "• Not enough data to detect category trends.\n"

        # Follow-up ideas
        response += "\n**💡 Want to explore more?** Try:\n"
        response += "• 'Show papers from 2023 in this field'\n"
        response += "• 'Which authors are most active in AI?'\n"

        return response


    def _generate_comparison_response(self, query: str, results: List[Dict]) -> str:
        """Generate comparison response for any topic."""
        if len(results) < 2:
            return f"I need at least 2 papers to make a comparison about '{query}'. Found only {len(results)} paper(s)."
        
        response = f"**Comparison of papers on '{query}':**\n\n"
        
        for i, res in enumerate(results[:3], 1):  # Compare top 3
            response += f"**Paper {i}:**\n"
            response += f"📄 **Title:** {res.get('title', 'N/A')}\n"
            response += f"👥 **Authors:** {res.get('author', 'N/A')[:100]}...\n"
            response += f"📅 **Published:** {res.get('published_date', 'N/A')}\n"
            response += f"📂 **Category:** {res.get('primary_category', 'N/A')}\n"
            response += f"🎯 **Relevance:** {res.get('similarity', 0):.1%}\n"
            
            # Brief summary
            summary = res.get('summary', 'No summary available')
            if len(summary) > 200:
                summary = summary[:200] + "..."
            response += f"📝 **Focus:** {summary}\n\n"
        
        # Analysis
        response += "**Analysis:**\n"
        categories = [res.get('primary_category', 'Unknown') for res in results[:3]]
        if len(set(categories)) > 1:
            response += f"• Different research areas: {', '.join(set(categories))}\n"
        else:
            response += f"• All papers are in the same category: {categories[0]}\n"
        
        return response


    def _generate_definition_response(self, query: str, results: List[Dict]) -> str:
        """Generate definition response for any topic."""
        if not results:
            return self._generate_no_results_response(query, "definition")
        
        top_result = results[0]
        
        response = f"**Definition/Explanation of '{query}':**\n\n"
        response += f"Based on: **{top_result.get('title', 'N/A')}**\n"
        response += f"*By {top_result.get('author', 'Unknown authors')}*\n"
        response += f"*Published: {top_result.get('published_date', 'Unknown')}*\n\n"
        
        summary = top_result.get('summary', 'No summary available')
        if summary and summary != 'No summary available':
            # Extract key definition sentences
            sentences = summary.split('. ')
            definition_sentences = []
            
            for sentence in sentences[:4]:  # First 4 sentences usually contain definitions
                if any(keyword in sentence.lower() for keyword in ['is', 'are', 'refers to', 'defined as', 'means']):
                    definition_sentences.append(sentence)
            
            if definition_sentences:
                definition = '. '.join(definition_sentences) + '.'
            else:
                definition = '. '.join(sentences[:2]) + '.'
                
            response += f"**Explanation:** {definition}\n\n"
        
        # Additional context
        if len(results) > 1:
            response += f"**Additional Context:**\n"
            response += f"Found {len(results)} papers on this topic across these areas:\n"
            
            categories = {}
            for res in results:
                cat = res.get('primary_category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]:
                response += f"• {cat}: {count} papers\n"
        
        return response



    def _generate_recommendation_response(self, query: str, results: List[Dict]) -> str:
        """Generate recommendation response for any topic."""
        if not results:
            return self._generate_no_results_response(query, "recommendation")
        
        response = f"**Top recommendations for '{query}':**\n\n"
        
        for i, res in enumerate(results[:5], 1):
            response += f"**{i}. {res.get('title', 'Untitled')}**\n"
            response += f"   👥 {res.get('author', 'Unknown authors')[:80]}...\n"
            response += f"   📅 {res.get('published_date', 'Unknown date')}\n"
            response += f"   📂 {res.get('primary_category', 'Unknown category')}\n"
            response += f"   🎯 Relevance: {res.get('similarity', 0):.1%}\n"
            
            # Brief description
            summary = res.get('summary', 'No summary available')
            if len(summary) > 150:
                summary = summary[:150] + "..."
            response += f"   📝 {summary}\n\n"
        
        response += "**Why these papers?**\n"
        response += "• Ranked by relevance to your query\n"
        response += "• Covers different aspects of the topic\n"
        response += "• Includes recent and influential work\n"
        
        return response



    def _generate_category_response(self, query: str, results: List[Dict]) -> str:
        """
        Extract category code from query and analyze matching papers.
        Show statistics + recent results.
        """
        import re
        # Match arXiv-style category like cs.LG, stat.ML, math.OC
        category_match = re.search(r"\b([a-z\-]+(?:\.[a-z\-]+))\b", query.lower())
        if not category_match:
            return "⚠️ I couldn't detect a valid arXiv category in your request. Try something like: `stats for cs.AI`"

        category_code = category_match.group(1)
        print(f"DEBUG: Extracted category '{category_code}'")

        if not hasattr(self, 'df') or self.df is None:
            return "⚠️ Data not loaded."

        df = self.df
        if 'primary_category' not in df.columns:
            return "⚠️ Primary category column is missing in the data."

        matches = df[df['primary_category'].str.lower() == category_code]

        if matches.empty:
            return f"No papers found in category `{category_code}`."

        # Statistics
        total = len(matches)
        years = matches['published_date'].dropna().astype(str).str[:4]
        year_dist = years.value_counts().sort_index()

        author_counter = Counter()
        for authors in matches['author'].dropna():
            for name in self._parse_authors(authors):
                author_counter[name] += 1

        response = f"**📂 Stats for Category `{category_code}`**\n\n"
        response += f"• Total papers: **{total}**\n"
        if not year_dist.empty:
            response += f"• Publication years: {year_dist.index.min()} – {year_dist.index.max()}\n"
        response += f"• Unique authors: {len(author_counter)}\n"

        # Top authors
        top_authors = author_counter.most_common(5)
        if top_authors:
            response += "\n**👥 Top Authors:**\n"
            for i, (name, count) in enumerate(top_authors, 1):
                response += f"{i}. {name} — {count} paper{'s' if count > 1 else ''}\n"

        # Recent papers
        recent = matches.sort_values("published_date", ascending=False).head(5)
        if not recent.empty:
            response += "\n**🆕 Most Recent Papers:**\n"
            for i, row in recent.iterrows():
                title = row['title'][:80] + "..." if len(row['title']) > 80 else row['title']
                authors = row['author'][:60] + "..." if len(row['author']) > 60 else row['author']
                date = self._format_pub_date(row.get('published_date'))
                response += f"- **{title}**\n  👥 {authors} | 📅 {date}\n"

        return response



    def _generate_default_response(self, query: str, results: List[Dict]) -> str:
        """Generate default search response."""
        top_result = results[0]
        
        response = f"Found {len(results)} articles related to '{query}'. "
        response += f"Here's the most relevant one:\n\n"
        response += f"**Title:** {top_result.get('title', 'N/A')}\n"
        response += f"**Authors:** {top_result.get('author', 'N/A')}\n"
        response += f"**Published:** {top_result.get('published_date', 'N/A')}\n"
        response += f"**Category:** {top_result.get('primary_category', 'N/A')}\n"
        response += f"**Similarity:** {top_result.get('similarity', 0):.3f}\n\n"
        
        summary = top_result.get('summary', 'No summary available')
        if len(summary) > 400:
            summary = summary[:400] + "..."
        response += f"**Summary:** {summary}"
        
        return response
        


    def _generate_no_results_response(self, query: str, intent: str) -> str:
        """Generate a helpful no-results response for any topic."""
        response = f"**No results found for '{query}'** 😔\n\n"
        
        response += "**Possible reasons:**\n"
        response += "• The topic might be too specific or new\n"
        response += "• Different terminology is used in academic papers\n"
        response += "• The search terms might need adjustment\n\n"
        
        response += "**💡 Suggestions to try:**\n"
        response += "• Use broader, more general terms\n"
        response += "• Try synonyms or related concepts\n"
        response += "• Check spelling and use standard academic terminology\n"
        response += "• Search for the main field first, then narrow down\n\n"
        
        # Intent-specific suggestions
        if intent == "author":
            response += "**For author searches:**\n"
            response += "• Try just the last name\n"
            response += "• Check for different name variations\n"
            response += "• Look for the author's institutional affiliation\n"
        elif intent == "recent":
            response += "**For recent papers:**\n"
            response += "• Try searching without date constraints first\n"
            response += "• The field might be established with older papers\n"
        elif intent == "definition":
            response += "**For definitions:**\n"
            response += "• Try searching for the broader field or category\n"
            response += "• Look for 'introduction' or 'survey' papers\n"
        
        response += "\n**Would you like me to help you refine your search?**"
        
        return response


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXiv Research Assistant")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    # parser.add_argument("--model", default="allenai/specter", help="Embedding model")
    parser.add_argument("--memory-limit", type=float, default=4.0, help="Memory limit in GB")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    
    args = parser.parse_args()
    
    try:
        chatbot = ArxivChatbot(
            openai_api_key=args.openai_key,
            model_name=args.model,
            max_memory_gb=args.memory_limit
        )
        
        if args.web:
            import streamlit as st
            chatbot.web_interface()
        else:
            chatbot.chat_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if 'chatbot' in locals():
            chatbot.cleanup()