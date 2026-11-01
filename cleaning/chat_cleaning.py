import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("stopwords")
nltk.download("wordnet")

df1 = pd.read_csv("data/chat/dataset_1.csv")
df2 = pd.read_csv("data/chat/dataset_2.csv")


# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# ============================================================
# DATASET 1
# ============================================================
print("Cleaning dataset_1...")
df1 = pd.read_csv("data/chat/dataset_1.csv")

df1 = df1.dropna(subset=["text", "toxicity_score"])
df1["text"] = df1["text"].apply(clean_text)
df1 = df1[df1["text"].str.strip() != ""]

# Rename ONLY for output
df1 = df1.rename(columns={
    "text": "chat",
    "toxicity_score": "score"
})

df1.to_csv("data/chat/dataset_1_cleaned.csv", index=False)
print(f"dataset_1 cleaned. Shape: {df1.shape}")

# ============================================================
# DATASET 2
# ============================================================
print("\nCleaning dataset_2...")
df2 = pd.read_csv("data/chat/dataset_2.csv")

df2 = df2.dropna(subset=["text", "toxicity_human_normalized"])
df2["text"] = df2["text"].apply(clean_text)
df2 = df2[df2["text"].str.strip() != ""]

# Rename ONLY for output
df2 = df2.rename(columns={
    "text": "chat",
    "toxicity_human_normalized": "score"
})

df2.to_csv("data/chat/dataset_2_cleaned.csv", index=False)
print(f"dataset_2 cleaned. Shape: {df2.shape}")

print("\nBoth datasets cleaned and saved to data/chat/")
