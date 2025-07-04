import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import argparse
from interface import main as streamlit_main
from chatbot import ArxivChatbot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_streamlit():
    """Run the Streamlit interface."""
    streamlit_main()

def run_chatbot_server(port: int = 8000):
    """Run the chatbot as a web server."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI()
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize chatbot
    chatbot = ArxivChatbot()
    
    @app.get("/search")
    async def search(query: str, k: int = 5):
        """Search endpoint."""
        results = chatbot.search_articles(query, k)
        response = chatbot.generate_response(query, results)
        return {"query": query, "results": results, "response": response}
    
    logger.info(f"Starting chatbot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arXiv Chatbot")
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
    
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        run_streamlit()
    else:
        run_chatbot_server(args.port)