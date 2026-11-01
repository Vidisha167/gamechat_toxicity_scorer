# GameChat Toxicity Scorer

An NLP-based toxicity detection module developed as part of a larger **Gaming Addiction Risk Detection System**. The module analyzes text messages exchanged in online gaming environments and estimates their toxicity using machine learning.

The predicted toxicity score serves as one of the inputs to a broader risk assessment framework that combines chat, voice, and behavioural signals.

## Overview

Online gaming communities often involve toxic communication, including harassment, abusive language, hate speech, and targeted insults. Identifying such interactions can help provide insights into unhealthy gaming behaviour and online interactions.

GameChat Toxicity Scorer uses Natural Language Processing (NLP) and machine learning to classify chat messages and estimate their toxicity.

### Key Features

- **Text Preprocessing:** Cleans chat messages through lowercasing, URL removal, punctuation and number removal, stopword removal, and lemmatization.
- **TF-IDF Feature Extraction:** Converts cleaned text into numerical feature vectors using TF-IDF vectorizers.
- **Dual-Model Prediction:** Uses two independently trained XGBoost classifiers to evaluate toxicity.
- **Confidence-Based Scoring:** Prioritizes predictions from the second model when its confidence exceeds a predefined threshold.
- **Keyword-Based Adjustment:** Uses a predefined list of toxicity-related keywords to adjust predictions when the models produce low scores.
- **Modular Integration:** Generates a chat toxicity score that can be incorporated into a larger multimodal gaming risk assessment system.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| NLP | NLTK |
| Feature Extraction | TF-IDF |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib, Pickle |
| Dataset | ToxiGen |

## System Architecture

```text
          Gaming Chat Message
                   |
                   v
           Text Preprocessing
                   |
        Lowercase, Cleaning,
       Stopword Removal, Lemmatization
                   |
                   v
           TF-IDF Vectorization
                   |
             +-----+-----+
             |           |
             v           v
          Model 1      Model 2
          XGBoost      XGBoost
             |           |
             v           v
          Score 1      Score 2
             |           |
             +-----+-----+
                   |
                   v
         Confidence-Based Selection
                   |
                   v
          Keyword-Based Adjustment
                   |
                   v
          Final Toxicity Score
                   |
                   v
       Multimodal Risk Assessment
```

## Model Pipeline

### 1. Dataset Preparation

The project uses the [ToxiGen dataset](https://huggingface.co/datasets/toxigen/toxigen-data), a dataset containing toxic and benign statements.

The dataset preparation script:

- Loads the annotated ToxiGen dataset.
- Handles missing human toxicity annotations.
- Applies Min-Max normalization to toxicity scores.
- Extracts the text and normalized toxicity score.
- Exports the processed data into a CSV file.

### 2. Text Preprocessing

Chat messages undergo preprocessing before feature extraction.

The preprocessing pipeline includes:

1. Converting text to lowercase.
2. Removing URLs.
3. Removing punctuation and special characters.
4. Removing numerical characters.
5. Eliminating English stopwords.
6. Applying WordNet lemmatization.

### 3. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to transform preprocessed chat messages into numerical representations.

Two separately fitted vectorizers are used to generate features for the corresponding trained models.

### 4. Toxicity Prediction

Two XGBoost classification models independently evaluate the processed message and generate toxicity probabilities.

The final scoring logic uses:

- The first model's prediction as the baseline.
- The second model's prediction when its confidence is at least 0.75.
- A keyword-based adjustment when predefined toxic expressions are detected but the predicted score is below 0.5.

This combines model-based predictions with a lightweight rule-based mechanism.

## Project Structure

```text
gamechat_toxicity_scorer/
│
├── cleaning/
│   └── Text cleaning and preprocessing
│
├── features/
│   └── TF-IDF feature extraction
│
├── training/
│   └── Model training scripts
│
├── models/
│   ├── dataset_1_tfidf.pkl
│   ├── dataset_2_tfidf.pkl
│   ├── dataset1_xgboost.pkl
│   └── dataset2_xgboost.pkl
│
├── predict.py
├── toxigen_dataset.py
├── view_npy.py
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Vidisha167/gamechat_toxicity_scorer.git

cd gamechat_toxicity_scorer
```

### 2. Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn xgboost joblib nltk datasets
```

### 3. Download NLTK Resources

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Prepare Model Artifacts

Place the trained model and vectorizer files inside a `models/` directory in the project root.

The prediction script expects the following files:

- `dataset_1_tfidf.pkl`
- `dataset_2_tfidf.pkl`
- `dataset1_xgboost.pkl`
- `dataset2_xgboost.pkl`

These artifacts must be generated using the corresponding training pipeline.

### 5. Run the Prediction Script

```bash
python predict.py
```

The script evaluates predefined test phrases and displays the predictions from both models alongside the final toxicity score.

## Example Test Cases

The prediction script includes examples covering different types of gaming communication:

| Input | Purpose |
|---|---|
| "ur trash get rekt noob" | Gaming-related insults |
| "good game guys" | Benign gaming communication |
| "i killed it today" | Context-dependent language |
| "women belong in the kitchen" | Potentially discriminatory language |

## Integration with the Larger System

GameChat Toxicity Scorer is designed as the chat-analysis component of a multimodal gaming addiction risk detection system.

The overall system considers three modalities:

- **Chat:** Toxicity and harmful communication patterns.
- **Voice:** Voice-based behavioural indicators.
- **Behaviour:** Gaming activity and usage patterns.

The chat toxicity output can be combined with the other modality-specific outputs to contribute to an overall risk assessment.

## Future Improvements

- Improve contextual understanding of sarcasm and gaming-specific language.
- Explore transformer-based NLP architectures.
- Expand training data with gaming-specific conversations.
- Evaluate model performance using precision, recall, F1-score, and ROC-AUC.
- Develop an API for real-time toxicity scoring.
- Improve robustness against false positives and context-dependent expressions.

## Disclaimer

This project is intended for research and educational purposes. Toxicity predictions are probabilistic estimates and should not be treated as definitive judgments about a user's behaviour or intent.