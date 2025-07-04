# arxiv-chatbot

scopus_chatbot_project/
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
