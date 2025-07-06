#!/usr/bin/env python3
"""
Performance test script for arXiv Chatbot.
Tests loading time, search speed, and memory usage.
"""

import time
import psutil
import gc
import sys
import os
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def timeout_handler(func, args=(), kwargs={}, timeout_duration=300, default=None):
    """Execute function with timeout."""
    result = [default]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        return default
    elif exception[0]:
        raise exception[0]
    else:
        return result[0]

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def test_data_loading():
    """Test data loading performance."""
    print("🔍 Testing data loading performance...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Test with limited data first
        from src.data_loader import ArxivDataLoader
        loader = ArxivDataLoader("data/processed/articles_clean.csv")
        df = loader.load_data(nrows=1000)  # Test with 1000 articles
        
        load_time = time.time() - start_time
        end_memory = get_memory_usage()
        memory_used = end_memory - start_memory
        
        print(f"✅ Data loading test:")
        print(f"   - Articles loaded: {len(df)}")
        print(f"   - Load time: {load_time:.2f} seconds")
        print(f"   - Memory used: {memory_used:.1f} MB")
        
        return df
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def test_model_loading():
    """Test model loading performance."""
    print("\n🔍 Testing model loading performance...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        from src.embedder import ArxivEmbedder
        embedder = ArxivEmbedder()
        
        load_time = time.time() - start_time
        end_memory = get_memory_usage()
        memory_used = end_memory - start_memory
        
        print(f"✅ Model loading test:")
        print(f"   - Model: {embedder.model_name}")
        print(f"   - Load time: {load_time:.2f} seconds")
        print(f"   - Memory used: {memory_used:.1f} MB")
        
        return embedder
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_search_performance(chatbot, test_queries):
    """Test search performance."""
    print("\n🔍 Testing search performance...")
    
    results = []
    for i, query in enumerate(test_queries, 1):
        try:
            start_time = time.time()
            
            # Test search
            search_results = chatbot.search_articles(query, k=5)
            
            search_time = time.time() - start_time
            
            # Test response generation
            start_time = time.time()
            response = chatbot.generate_response(query, search_results)
            response_time = time.time() - start_time
            
            results.append({
                'query': query,
                'search_time': search_time,
                'response_time': response_time,
                'total_time': search_time + response_time,
                'results_count': len(search_results)
            })
            
            print(f"   Query {i}: '{query[:50]}...'")
            print(f"     - Search: {search_time:.3f}s")
            print(f"     - Response: {response_time:.3f}s")
            print(f"     - Total: {search_time + response_time:.3f}s")
            print(f"     - Results: {len(search_results)}")
            
        except Exception as e:
            print(f"   ❌ Query {i} failed: {e}")
            continue
    
    if results:
        # Calculate averages
        avg_search_time = sum(r['search_time'] for r in results) / len(results)
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        avg_total_time = sum(r['total_time'] for r in results) / len(results)
        
        print(f"\n📊 Search performance summary:")
        print(f"   - Average search time: {avg_search_time:.3f}s")
        print(f"   - Average response time: {avg_response_time:.3f}s")
        print(f"   - Average total time: {avg_total_time:.3f}s")
    
    return results

def test_chatbot_initialization():
    """Test full chatbot initialization."""
    print("\n🔍 Testing full chatbot initialization...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Use timeout wrapper for initialization
        from src.chatbot import ArxivChatbot
        
        def init_chatbot():
            return ArxivChatbot()
        
        chatbot = timeout_handler(init_chatbot, timeout_duration=300, default=None)
        
        if chatbot is None:
            print("❌ Chatbot initialization timed out (5 minutes)")
            return None
        
        init_time = time.time() - start_time
        end_memory = get_memory_usage()
        memory_used = end_memory - start_memory
        
        print(f"✅ Chatbot initialization test:")
        print(f"   - Articles loaded: {len(chatbot.df)}")
        print(f"   - Initialization time: {init_time:.2f} seconds")
        print(f"   - Memory used: {memory_used:.1f} MB")
        
        return chatbot
        
    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        return None

def main():
    """Run all performance tests."""
    print("🚀 arXiv Chatbot Performance Test")
    print("=" * 50)
    
    # System info
    try:
        memory = psutil.virtual_memory()
        print(f"System memory: {memory.total / (1024**3):.1f} GB total")
        print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        print(f"Memory usage: {memory.percent}%")
    except:
        print("System memory info not available")
    print()
    
    # Test individual components
    df = test_data_loading()
    embedder = test_model_loading()
    
    # Test full chatbot
    chatbot = test_chatbot_initialization()
    
    if chatbot:
        # Test search performance
        test_queries = [
            "machine learning",
            "deep learning neural networks",
            "artificial intelligence",
            "computer vision",
            "natural language processing"
        ]
        
        search_results = test_search_performance(chatbot, test_queries)
        
        # Memory cleanup
        gc.collect()
        final_memory = get_memory_usage()
        print(f"\n🧹 Final memory usage: {final_memory:.1f} MB")
        
        # Performance assessment
        print("\n📈 Performance Assessment:")
        if search_results:
            avg_total = sum(r['total_time'] for r in search_results) / len(search_results)
            if avg_total < 1.0:
                print("   ✅ EXCELLENT: Response time < 1 second")
            elif avg_total < 3.0:
                print("   ✅ GOOD: Response time < 3 seconds")
            elif avg_total < 5.0:
                print("   ⚠️  ACCEPTABLE: Response time < 5 seconds")
            else:
                print("   ❌ SLOW: Response time > 5 seconds")
        
        print("\n🎯 Recommendations:")
        print("   - If response time > 3s: Consider reducing dataset size")
        print("   - If memory usage > 4GB: Consider using fewer articles")
        print("   - For production: Use server mode with --optimize flag")
    
    print("\n✅ Performance test completed!")

if __name__ == "__main__":
    main() 