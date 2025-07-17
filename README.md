arxiv_chatbot_project/
arxiv_env\Scripts\activate

# arxiv-chatbot

Assistant de recherche avancé pour explorer les articles scientifiques d'arXiv avec interface web, recherche sémantique, filtres puissants et réponses conversationnelles.

---

## Sommaire

1. [Présentation](#présentation)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Préparation des données](#préparation-des-données)
5. [Lancement de l'application](#lancement-de-lapplication)
6. [Utilisation](#utilisation)
7. [Structure du projet](#structure-du-projet)
8. [Configuration avancée](#configuration-avancée)
9. [Dépannage](#dépannage)
10. [Commandes utiles](#commandes-utiles)
11. [Contact](#contact)

---

## 1. Présentation

**arxiv-chatbot** est un assistant de recherche interactif pour explorer, filtrer et analyser les articles scientifiques d'arXiv. Il propose :

- Une interface web moderne (Streamlit)
- Un moteur de recherche sémantique rapide (FAISS)
- Des filtres avancés (auteur, année, catégorie, similarité…)
- Des réponses conversationnelles (LLM OpenAI ou modèles locaux)
- Des outils de nettoyage et de préparation de données pour de grands jeux de données

---

## 2. Prérequis

- **Python 3.8+** (recommandé : 3.10 ou 3.11)
- **Git** (pour cloner le projet)
- **Connexion internet** (pour télécharger les modèles et dépendances)
- **(Optionnel) Clé OpenAI** pour des réponses conversationnelles avancées

---

## 3. Installation

### a. Cloner le projet

```bash
git clone https://github.com/Ouss-tagh-dev/arxiv-chatbot.git
cd arxiv-chatbot
```

### b. Créer et activer un environnement virtuel

Sous **Windows** :

```bash
python -m venv arxiv_env
arxiv_env\Scripts\activate
```

Sous **Linux/Mac** :

```bash
python3 -m venv arxiv_env
source arxiv_env/bin/activate
```

### c. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Préparation des données

### a. Nettoyer les données arXiv

Si vous partez de données brutes (CSV), lancez :

```bash
python src/data_cleaning.py --input data/raw/articles.csv --output data/processed/articles_clean.csv --deep-clean --use-spacy --remove-stopwords --lemmatize --sample-size 20000
```

- **--deep-clean** : nettoyage NLP avancé
- **--use-spacy** : utilise spaCy pour le traitement linguistique
- **--sample-size** : (optionnel) pour créer un sous-échantillon

### b. Générer les embeddings (vecteurs sémantiques)

```bash
python generate_index.py --data data/processed/articles_clean.csv --output data/embeddings/ --text_field summary
```

Cela crée les fichiers d'index FAISS nécessaires à la recherche rapide.

---

## 5. Lancement de l'application

### a. Interface web (recommandé)

```bash
streamlit run src/chatbot.py -- --web
```

ou :

```bash
python -m streamlit run src/chatbot.py -- --web
```

### b. Mode terminal (chat CLI)

```bash
python src/chatbot.py
```

---

## 6. Utilisation

- **Recherche sémantique** : tapez une question ou un sujet scientifique
- **Filtres** : affinez par auteur, année, catégorie, similarité, etc.
- **Onglet Chat** : conversation avec l’assistant
- **Onglet Résultats** : liste détaillée des articles trouvés
- **Statistiques** : analyse rapide des résultats (catégories, années, auteurs…)

---

## 7. Structure du projet

```
arxiv-chatbot/
│
├── README.md                # Ce guide complet
├── requirements.txt         # Dépendances Python
├── generate_index.py        # Génération des embeddings FAISS
├── get_column.py            # Script utilitaire (extraction de colonnes)
├── test_performance.py      # Script de test de performance
├── Enonce.md                # Sujet ou consignes du projet
├── projet_Master_Python_24_25_250606_111034.pdf # Document projet
│
├── data/                    # Données et artefacts intermédiaires
│   ├── raw/                 # Données brutes (CSV, JSON)
│   │   └── articles.csv
│   ├── processed/           # Données nettoyées (plusieurs CSV, stats JSON)
│   │   ├── articles_clean.csv
│   │   ├── comprehensive_stats.json
│   │   ├── enhanced_cleaning_stats.json
│   │   ├── essential_articles_clean.csv
│   │   └── text_only_articles_clean.csv
│   ├── embeddings/          # Index FAISS et métadonnées
│   │   ├── faiss_index_all-MiniLM-L6-v2.index
│   │   └── faiss_metadata_all-MiniLM-L6-v2.pkl
│   ├── logs/                # Logs d'import ou de traitement (ex: daily_fetch.log)
│   ├── cache/               # (Prévu) Cache temporaire (actuellement vide)
│   ├── cleaned/             # (Prévu) Données nettoyées alternatives (vide)
│   └── interim/             # (Prévu) Fichiers intermédiaires (vide)
│
├── notebooks/               # Notebooks d'exploration et de traitement
│   ├── cleaning_data.ipynb
│   ├── extract_data.ipynb
│   └── old_extract-code.ipynb
│
├── src/                     # Code source Python (modulaire)
│   ├── __init__.py
│   ├── chatbot.py           # Moteur de dialogue principal (Streamlit/CLI)
│   ├── cleaner.py           # Nettoyage avancé des données
│   ├── data_cleaning.py     # Pipeline de nettoyage (script principal)
│   ├── data_loader.py       # Chargement et gestion mémoire des données
│   ├── embedder.py          # Génération des embeddings sémantiques
│   └── search_engine.py     # Recherche sémantique (FAISS)
│
├── logs/                    # (Prévu) Logs d’exécution (vide)
├── stats/                   # (Prévu) Statistiques additionnelles (vide)
├── models/                  # (Prévu) Modèles NLP téléchargés (vide)
└── .gitignore
```

**Remarques :**

- Les dossiers `cache/`, `cleaned/`, `interim/`, `logs/`, `stats/`, `models/` sont prévus pour des usages futurs ou pour organiser le flux de données, même s’ils sont vides par défaut.
- Le fichier `README.md` est à la racine et sert de guide principal.
- Tous les scripts et modules sont dans `src/` pour une meilleure maintenabilité.
- Les notebooks sont dédiés à l’exploration, à la reproductibilité scientifique et à l’extraction/traitement des données (extraction, nettoyage, analyse exploratoire, etc.).

---

## 8. Configuration avancée

- **Clé OpenAI** : pour des réponses plus naturelles, ajoutez `--openai-key VOTRE_CLE` lors du lancement
- **Limite mémoire** : ajustez avec `--memory-limit 4.0` (en Go)
- **Modèle d'embedding** : changez avec `--model all-MiniLM-L6-v2` ou un autre modèle compatible

---

## 9. Dépannage

- **Problème de mémoire** : le chatbot gère automatiquement la mémoire, mais vous pouvez réduire la taille des données ou la limite mémoire si besoin.
- **Pas de résultats** : vérifiez que les fichiers d’index sont bien générés et que les données sont propres.
- **Erreur OpenAI** : assurez-vous d’avoir une clé valide et une connexion internet.
- **Dépendances manquantes** : relancez `pip install -r requirements.txt`.

---

## 10. Commandes utiles

```bash
# Nettoyer les données
python src/data_cleaning.py --input data/raw/articles.csv --output data/processed/articles_clean.csv --deep-clean --use-spacy

# Générer les embeddings
python generate_index.py --data data/processed/articles_clean.csv --output data/embeddings/ --text_field summary

# Lancer l’interface web
streamlit run src/chatbot.py -- --web

# Lancer le chat en terminal
python src/chatbot.py
```

---

## 👥 Contributeurs

| Membre            | GitHub                                               |
| ----------------- | -----------------------------------------------------|
| Oussama TAGHLAOUI | [ouss-tagh-dev](https://github.com/ouss-tagh-dev)    |
| Abd'allah Ismaili | [AbdallahIsmaili](https://github.com/AbdallahIsmaili)|
| Sanaa AZZA        | [sanaaazza](https://github.com/sanaaazza)            |