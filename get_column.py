import pandas as pd

# Afficher la liste des colonnes
# print(pd.read_csv("data/processed/articles_clean.csv", nrows=1).columns.tolist())

# Afficher la colonne 'category' des 3 premières lignes
category = pd.read_csv("data/processed/articles_clean.csv", usecols=["category"], nrows=3)
print(category)
