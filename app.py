import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.interface import main as streamlit_main
from src.chatbot import ArxivChatbot
import logging
import argparse
import gc

# Configure logging for performance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_for_performance():
    """Optimize system settings for better performance on low-end machines."""
    import psutil
    
    # Set lower priority for the process (Windows)
    try:
        import win32api
        import win32process
        import win32con
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to below normal for better system performance")
    except ImportError:
        logger.info("win32api not available, skipping priority adjustment")
    
    # Log memory info
    memory = psutil.virtual_memory()
    logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
    logger.info(f"Memory usage: {memory.percent}%")

def run_streamlit():
    """Run the Streamlit interface with optimizations."""
    # Optimize before starting
    optimize_for_performance()
    
    # Force garbage collection
    gc.collect()
    
    logger.info("Starting Streamlit interface with optimizations...")
    streamlit_main()

def run_chatbot_server(port: int = 8000):
    """Run the chatbot as a web server with optimizations."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # Optimize before starting
    optimize_for_performance()
    
    app = FastAPI(title="arXiv Chatbot API")
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize chatbot with error handling
    try:
        logger.info("Initializing chatbot...")
        chatbot = ArxivChatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise
    
    @app.get("/search")
    async def search(query: str, k: int = 5):
        """Search endpoint with error handling."""
        try:
            results = chatbot.search_articles(query, k)
            response = chatbot.generate_response(query, results)
            return {"query": query, "results": results, "response": response}
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": str(e), "query": query}
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "articles_loaded": len(chatbot.df)}
    
    logger.info(f"Starting chatbot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arXiv Chatbot - Optimized for Performance")
    parser.add_argument(
        "--mode",
        choices=["streamlit", "server"],
        default="streamlit",
        help="Run mode (default: streamlit)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for server mode (default: 8000)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable performance optimizations"
    )
    
    args = parser.parse_args()
    
    if args.optimize:
        logger.info("Performance optimizations enabled")
        optimize_for_performance()
    
    try:
        if args.mode == "streamlit":
            run_streamlit()
        else:
            run_chatbot_server(args.port)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)