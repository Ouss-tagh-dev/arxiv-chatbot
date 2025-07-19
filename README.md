# arxiv-chatbot

Assistant de recherche avancé pour explorer les articles scientifiques d'arXiv avec interface web, recherche sémantique, filtres puissants et réponses conversationnelles.

<p align="center">
  <img src="images/How many papers are there in machine learning.png" alt="ML Paper Count" width="45%"/>
  <img src="images/written by Nathaniel Eldredge.png" alt="Written by Nathaniel Eldredge" width="45%"/>
</p>
<p align="center">
  <img src="images/Can machine learning identify language varieties.png" alt="ML Language Varieties" width="45%"/>
  <img src="images/Can machine learning identify language varieties Result.png" alt="ML Language Varieties Result" width="45%"/>
</p>

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Prérequis](#2-prérequis)
3. [Installation](#3-installation)
4. [Préparation des données](#4-préparation-des-données)
5. [Lancement de l'application](#5-lancement-de-lapplication)
6. [Utilisation](#6-utilisation)
7. [Exemples de requêtes efficaces](#7-exemples-de-requêtes-efficaces)
8. [Structure du projet](#8-structure-du-projet)
9. [Scripts d’importation arXiv](#9-scripts-dimportation-arxiv)
10. [Configuration avancée](#10-configuration-avancée)
11. [Dépannage](#11-dépannage)
12. [Commandes utiles](#12-commandes-utiles)
13. [Contributeurs](#13-contributeurs)

---

## 1. Présentation

**arxiv-chatbot** est un assistant de recherche interactif pour explorer, filtrer et analyser les articles scientifiques d'arXiv. Il propose :

- Une interface web moderne (Streamlit)
- Un moteur de recherche sémantique rapide (FAISS)
- Des filtres avancés (auteur, année, catégorie, similarité...)
- Des outils de nettoyage et de préparation de données pour de grands jeux de données
- Des scripts de récupération des données depuis arXiv

---

## 2. Prérequis

- **Python 3.8+** (recommandé : 3.10 ou 3.11)
- **Git** (pour cloner le projet)
- **Connexion internet** (pour télécharger les modèles et dépendances)

---

## 3. Installation

### a. Cloner le projet

```bash
git clone https://github.com/Ouss-tagh-dev/arxiv-chatbot.git
cd arxiv-chatbot
```

### b. Créer et activer un environnement virtuel

#### Sous **Windows** :

```bash
python -m venv arxiv_env
arxiv_env\Scripts\activate
```

#### Sous **Linux/Mac** :

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

```bash
python src/data_cleaning.py --input data/raw/articles.csv --output data/processed/articles_clean.csv --deep-clean --use-spacy --remove-stopwords --lemmatize --sample-size 20000
```

**Paramètres principaux** :

- `--input` : chemin du fichier CSV brut contenant les articles arXiv
- `--output` : chemin du fichier de sortie nettoyé
- `--deep-clean` : active un nettoyage avancé (suppression des symboles, normalisation, etc.)
- `--use-spacy` : utilise la bibliothèque spaCy pour l'analyse linguistique (tokenisation, POS, etc.)
- `--remove-stopwords` : supprime les mots vides ("le", "de", "and"...) pour améliorer la qualité des embeddings
- `--lemmatize` : réduit les mots à leur lemme (forme de base) ex : "running" devient "run"
- `--sample-size` : limite le nombre de lignes à traiter (utile pour tests rapides ou faible RAM)

### b. Générer les embeddings (vecteurs sémantiques)

```bash
python src/generate_index.py --data data/processed/articles_clean.csv --output data/embeddings/ --text_field summary
```

**Paramètres principaux** :

- `--data` : fichier CSV nettoyé à indexer
- `--output` : répertoire où stocker l'index FAISS et les métadonnées
- `--text_field` : colonne du fichier CSV contenant le texte à transformer en vecteurs (ex : "summary")

---

## 5. Lancement de l'application

### a. Interface web (mode graphique)

```bash
streamlit run src/chatbot.py -- --web
```

ou :

```bash
python -m streamlit run src/chatbot.py -- --web
```

### b. Mode terminal (chat en ligne de commande)

```bash
python src/chatbot.py
```

---

## 6. Utilisation

- Tapez une question scientifique librement.
- Utilisez les filtres par catégorie, date, auteur ou similarité.
- Utilisez l’onglet **Chat** pour interagir.
- Consultez l’onglet **Résultats** pour les articles pertinents.

---

## 7. Exemples de requêtes efficaces

| Type                | Exemple de prompt                       |
| ------------------- | --------------------------------------- |
| **Statistiques**    | How many papers in astro-ph.EP in 2013? |
| **Par auteur**      | papers by David Graus                   |
| **Récentes**        | recent articles in AI in 2023           |
| **Résumé**          | Summarize "Friends of Hot Jupiters I"   |
| **Comparaison**     | Compare data science and data analytics |
| **Tendances**       | What are trending categories in 2021?   |
| **Définition**      | Define Machine Learning Algorithms      |
| **Recommandations** | Recommend best papers in quantum optics |
| **Par catégorie**   | List papers in category astro-ph.CO     |

---

## 8. Structure du projet

```bash
arxiv-chatbot/
|
├── README.md                          # Ce guide
├── requirements.txt                   # Liste des packages Python
├── Enonce.md                          # Consignes du projet
├── projet_Master_Python_24_25.pdf    # Sujet fourni
|
├── data/                              # Données
│   ├── raw/                           # Données brutes
│   ├── processed/                     # Données nettoyées
│   ├── embeddings/                    # Index FAISS
│   ├── logs/                          # Logs
│   └── cache/                         # Cache futur
|
├── notebooks/                         # Notebooks d’extraction
│   ├── bulk_arxiv_to_mongodb.ipynb   # Import massif historique
│   └── daily_arxiv_updater.ipynb     # Mise à jour quotidienne
|
├── src/                               # Code source principal
│   ├── chatbot.py                     # Interface conversationnelle (CLI & web), gestion dialogue et recherche
│   ├── data_cleaning.py               # Script de nettoyage avancé des données arXiv (prétraitement NLP)
│   ├── cleaner.py                     # Module principal de nettoyage et prétraitement NLP (normalisation, lemmatisation, etc.)
│   ├── data_loader.py                 # Chargement robuste des données CSV, gestion mémoire et types
│   ├── embedder.py                    # Génération d'embeddings sémantiques pour les articles (batch & temps réel)
│   ├── generate_index.py              # Génération de l’index FAISS et des métadonnées pour la recherche
│   └── search_engine.py               # Moteur de recherche sémantique FAISS optimisé (similarité, reranking, GPU)
|
└── .gitignore
```

---

## 9. Scripts d’importation arXiv

### a. `bulk_arxiv_to_mongodb.ipynb`

- **But** : import massif historique depuis arXiv.
- **Stockage** : MongoDB (optimisé pour gros volumes).
- **Multithread** : via `MAX_THREADS` pour parallélisation.

### b. `daily_arxiv_updater.ipynb`

- **But** : synchronisation quotidienne automatique.
- **Filtrage** : télécharge uniquement les nouveaux articles.
- **Nettoyage** : évite les doublons dans `articles.csv`.

---

## 10. Configuration avancée

- `--memory-limit` : limite mémoire pour traitement lourd.
- `--model` : changer le modèle de vectorisation (par ex. `all-MiniLM-L6-v2`, `paraphrase-MiniLM`, etc.)

---

## 11. Dépannage

- **Erreur de mémoire** : réduire `--sample-size` ou augmenter RAM.
- **Aucun résultat** : s'assurer que les embeddings ont bien été générés.
- **Erreur de dépendance** : exécuter `pip install -r requirements.txt`.

---

## 12. Commandes utiles

```bash
# Nettoyer les données
python src/data_cleaning.py --input data/raw/articles.csv --output data/processed/articles_clean.csv --deep-clean --use-spacy --remove-stopwords --lemmatize

# Générer les embeddings
python src/generate_index.py --data data/processed/articles_clean.csv --output data/embeddings/ --text_field summary

# Lancer l’application Streamlit
streamlit run src/chatbot.py -- --web

# Lancer en mode terminal
python src/chatbot.py
```

---

## 13. Contributeurs

| Nom               | GitHub                                                 |
| ----------------- | ------------------------------------------------------ |
| Oussama TAGHLAOUI | [@ouss-tagh-dev](https://github.com/ouss-tagh-dev)     |
| Abd'allah Ismaili | [@AbdallahIsmaili](https://github.com/AbdallahIsmaili) |
| Sanaa AZZA        | [@sanaaazza](https://github.com/sanaaazza)             |
