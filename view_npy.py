import numpy as np
import pandas as pd
import joblib

# Load features
X = np.load("data/chat/chat_features.npy")

# Load vectorizer to get word names
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

feature_names = vectorizer.get_feature_names_out()

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)

# (Optional but VERY useful) add original chat text
chat_data = pd.read_csv("data/chat/chat_cleaned.csv")
df["chat"] = chat_data["chat"]

# Save to CSV
df.to_csv("data/chat/chat_features_readable.csv", index=False)

print("CSV created successfully!")