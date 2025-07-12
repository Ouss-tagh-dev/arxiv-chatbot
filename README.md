# arxiv-chatbot

arxiv_chatbot_project/
│
├── data/                         # Données brutes et nettoyées
│   ├── raw/                      # Données extraites brutes (CSV, JSON)
│   ├── processed/ # Données nettoyées prêtes à l'usage
│   ├── logs/ # Logs de l'application
│   └── embeddings/              # Vecteurs sémantiques des résumés
│
├── notebooks/                   # Notebooks d'exploration et développement
│   ├── extract_data.ipynb
│   ├── cleaning_data.ipynb
│   └── exploration.ipynb
│
├── src/                         # Code source Python organisé
│   ├── __init__.py
│   ├── data_loader.py           # Extraction / chargement des données
│   ├── cleaner.py               # Nettoyage des données
│   ├── embedder.py              # Vectorisation sémantique des résumés
│   ├── search_engine.py         # Indexation et recherche via FAISS/ChromaDB
│   ├── interface.py             # Interface utilisateur (Streamlit / Flask)
│   └── chatbot.py               # Moteur de dialogue intelligent (optionnel)
│
├── models/                      # Modèles NLP (transformers téléchargés si besoin)
│
├── requirements.txt             # Dépendances Python
├── app.py                       # Script principal (exécution interface web)
├── README.md                    # Explication du projet
└── .gitignore




### commands

python -m venv arxiv_env
source arxiv_env/bin/activate  
arxiv_env\Scripts\activate 

py src/data_cleaning.py --input data/raw/articles.csv --output data/processed/articles_clean.csv

<!-- py generate_index.py --data data/processed/articles_clean.csv --output data/embeddings/ --text_field summary -->
py generate_index.py --data data/processed/articles_clean.csv --output data/embeddings/ --text_field summary --quick --nrows 20000

streamlit run src/chatbot.py
streamlit run src/chatbot.py -- --web
python -m streamlit run src/chatbot.py -- --web