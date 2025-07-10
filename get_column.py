import pandas as pd
print(pd.read_csv("data/processed/articles_clean.csv", nrows=1).columns.tolist())