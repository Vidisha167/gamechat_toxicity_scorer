import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# DATASET 1
# ============================================================
print("Processing dataset_1 TF-IDF...")

df1 = pd.read_csv("data/chat/dataset_1_cleaned.csv")
texts1 = df1["chat"].astype(str)

tfidf1 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X1 = tfidf1.fit_transform(texts1)

sp.save_npz("data/chat/dataset_1_features.npz", X1)

with open("models/dataset_1_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf1, f)

print(f"dataset_1 TF-IDF done. Shape: {X1.shape}")

# ============================================================
# DATASET 2
# ============================================================
print("\nProcessing dataset_2 TF-IDF...")

df2 = pd.read_csv("data/chat/dataset_2_cleaned.csv")
texts2 = df2["chat"].astype(str)

tfidf2 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X2 = tfidf2.fit_transform(texts2)

sp.save_npz("data/chat/dataset_2_features.npz", X2)

with open("models/dataset_2_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf2, f)

print(f"dataset_2 TF-IDF done. Shape: {X2.shape}")

print("\nDone! Load features later using:")
print('X = sp.load_npz("data/chat/dataset_1_features.npz")')