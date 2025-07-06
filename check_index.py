#!/usr/bin/env python3
"""
Quick script to check if FAISS index exists and diagnose loading issues.
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if required files exist."""
    print("🔍 Checking required files...")
    
    files_to_check = [
        "data/processed/articles_clean.csv",
        "data/embeddings/arxiv_faiss_index.index",
        "data/embeddings/arxiv_metadata.pkl"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path} ({size:,} bytes)")
        if not exists:
            all_exist = False
    
    return all_exist

def check_data_sample():
    """Check if we can load a small sample of data."""
    print("\n🔍 Testing data loading...")
    
    try:
        import pandas as pd
        df = pd.read_csv("data/processed/articles_clean.csv", nrows=10)
        print(f"✅ Data loading test: {len(df)} rows loaded")
        print(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def check_model():
    """Check if sentence transformer model can be loaded."""
    print("\n🔍 Testing model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        print(f"✅ Model loading test: {model.model_name}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all checks."""
    print("🚀 arXiv Chatbot - File and System Check")
    print("=" * 50)
    
    # Check files
    files_ok = check_files()
    
    # Check data loading
    data_ok = check_data_sample()
    
    # Check model loading
    model_ok = check_model()
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Files: {'✅' if files_ok else '❌'}")
    print(f"   Data: {'✅' if data_ok else '❌'}")
    print(f"   Model: {'✅' if model_ok else '❌'}")
    
    if files_ok and data_ok and model_ok:
        print("\n✅ All checks passed! You can run the chatbot.")
        print("   Command: python -m streamlit run app.py -- --mode streamlit")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        
        if not files_ok:
            print("\n💡 To generate the FAISS index:")
            print("   1. Make sure you have the data file")
            print("   2. Run: python generate_index.py")
        
        if not data_ok:
            print("\n💡 Data file issue:")
            print("   - Check if data/processed/articles_clean.csv exists")
            print("   - Verify the file is not corrupted")
        
        if not model_ok:
            print("\n💡 Model issue:")
            print("   - Install sentence-transformers: pip install sentence-transformers")
            print("   - Check internet connection for model download")

if __name__ == "__main__":
    main() 