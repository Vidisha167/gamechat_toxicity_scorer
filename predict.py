import os
import pickle
import joblib
import scipy.sparse as sp

# ============================================================
# PATH SETUP (self-contained)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script is
MODELS_DIR = os.path.join(BASE_DIR, "models")

tfidf1_path = os.path.join(MODELS_DIR, "dataset_1_tfidf.pkl")
tfidf2_path = os.path.join(MODELS_DIR, "dataset_2_tfidf.pkl")
model1_path = os.path.join(MODELS_DIR, "dataset1_xgboost.pkl")
model2_path = os.path.join(MODELS_DIR, "dataset2_xgboost.pkl")

# ============================================================
# LOAD VECTORZIERS AND MODELS
# ============================================================
with open(tfidf1_path, "rb") as f:
    tfidf1 = pickle.load(f)
with open(tfidf2_path, "rb") as f:
    tfidf2 = pickle.load(f)

model1 = joblib.load(model1_path)
model2 = joblib.load(model2_path)

# ============================================================
# CLEAN TEXT FUNCTION (include your cleaning logic here)
# ============================================================
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ============================================================
# PREDICT TOXICITY
# ============================================================
TOXIC_KEYWORDS = [
    "kys", "kill yourself", "hate", "idiot", "moron", "stupid",
    "worthless", "garbage", "trash", "noob", "rekt", "worst",
    "nobody likes you", "quit", "useless", "shut up"
]

def predict_toxicity(text, tfidf, model):
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])

    if features.nnz == 0:
        return 0.0

    proba = model.predict_proba(features)[:, 1]
    return round(float(proba[0]), 4)

def final_chat_score(text):
    score1 = predict_toxicity(text, tfidf1, model1)
    score2 = predict_toxicity(text, tfidf2, model2)

    # Keyword boost
    text_lower = text.lower()
    keyword_hit = any(kw in text_lower for kw in TOXIC_KEYWORDS)

    # Only trust score2 if very confident
    if score2 >= 0.75:
        final = round(max(score1, score2), 4)
    else:
        final = round(score1, 4)

    # Boost if keyword hit but model missed
    if keyword_hit and final < 0.5:
        final = round(max(final + 0.3, 0.5), 4)

    return score1, score2, final

# ============================================================
# TEST PHRASES
# ============================================================
test_phrases = [
    "women belong in the kitchen",
    "all muslims are terrorists",
    "ur trash get rekt noob",
    "good game guys",
    "i killed it today",
]

print(f"{'Phrase':<35} {'Score 1':>10} {'Score 2':>10} {'Final':>10}")
print("=" * 67)

for phrase in test_phrases:
    s1, s2, final = final_chat_score(phrase)
    print(f"{phrase:<35} {s1:>10.4f} {s2:>10.4f} {final:>10.4f}")