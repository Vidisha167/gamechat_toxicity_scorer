import pandas as pd
from datasets import load_dataset

# Load ToxiGen dataset from HuggingFace
dataset = load_dataset("toxigen/toxigen-data", name="annotated")

# Convert to Pandas DataFrame
df = dataset["train"].to_pandas()

# Handle any NaN values
df["toxicity_human"] = df["toxicity_human"].fillna(df["toxicity_human"].mean())

# Min-Max Normalization → maps scores to [0, 1]
min_val = df["toxicity_human"].min()
max_val = df["toxicity_human"].max()

df["toxicity_human_normalized"] = (df["toxicity_human"] - min_val) / (max_val - min_val)

# Keep only text and normalized score
df = df[["text", "toxicity_human_normalized"]]

# Preview the result
print(df.head(10))
print("\nShape:", df.shape)

# Save to CSV
df.to_csv("subtle_toxic_chat.csv", index=False)
print("\nSaved to subtle_toxic_chat.csv")