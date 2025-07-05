# Énoncé du Projet : Chatbot basé sur les données arXiv

## Objectif du projet
Développer un chatbot intelligent capable d'interagir avec l'utilisateur en langage naturel afin de répondre à des questions portant sur des publications scientifiques issues de arXiv. Le chatbot exploitera des techniques de NLP et d'indexation sémantique pour offrir des réponses pertinentes, synthétiques et contextualisées.

## Besoins fonctionnels
Le système devra permettre :
- L'extraction et la gestion automatisée de données bibliographiques depuis arXiv.
- Une recherche intelligente basée sur la compréhension des intentions (intent) et entités dans les requêtes utilisateurs.
- La génération de réponses personnalisées, sous forme de textes, résumés ou visualisations.
- Une interface utilisateur fluide, accessible et facile à prendre en main.

## Étapes de réalisation

### a. Extraction des données arXiv
- Extraction automatique via l'API arXiv ou téléchargement des jeux de données publics.
- Données ciblées : Titre, résumé, auteurs, affiliation, année, identifiant arXiv, mots-clés.

### b. Nettoyage et stockage
- Nettoyage des données avec Pandas :
  - Suppression des doublons, normalisation des auteurs et affiliations.
  - Traitement des caractères spéciaux et valeurs manquantes.
- Stockage local : Structuration en tables relationnelles (auteurs, articles, affiliations, etc.).

### c. Indexation sémantique des résumés
- Utilisation de Sentence Transformers (BERT, MiniLM, etc.) pour vectoriser les résumés.
- Création d'un index vectoriel via FAISS ou ChromaDB pour recherche sémantique rapide.
- L'index permet de retrouver les articles les plus proches d'une requête textuelle.

### d. Interface utilisateur simple et fluide
- Web app légère avec Streamlit, Gradio ou Flask.
- Fonctionnalités :
  - Zone de saisie des questions.
  - Affichage clair des résultats.
  - Option de filtrage par année, auteur, etc.
  - Visualisations interactives (Plotly, Altair, etc.).

## Livrables attendus
- Script d'extraction et de traitement des données arXiv.
- Base de données structurée avec contenu nettoyé.
- Moteur de recherche sémantique.
- Application web fonctionnelle avec chatbot.
- Documentation technique et guide utilisateur. 