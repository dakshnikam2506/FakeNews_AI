import os
import re

import joblib
import numpy as np


# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# Load the trained models and TF-IDF vectorizer
tfidf = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

logistic_regression = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

naive_bayes = joblib.load(
    os.path.join(MODEL_DIR, "naive_bayes.pkl")
)

linear_svm = joblib.load(
    os.path.join(MODEL_DIR, "linear_svm.pkl")
)


# Project labels: 0 = FAKE, 1 = REAL
LABEL_MAP = {
    0: "FAKE",
    1: "REAL",
}


def preprocess_text(text):
    """Clean news text before feature extraction."""

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    # Keep alphabetic characters and normalize whitespace
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def convert_label(label):
    """Convert a numeric class label into its text equivalent."""

    return LABEL_MAP.get(int(label), "UNKNOWN")


def predict_news(text):
    """Classify news using Logistic Regression, Naive Bayes, and SVM."""

    if not text or not str(text).strip():
        raise ValueError("News text cannot be empty.")

    # Preprocess the input and generate TF-IDF features
    cleaned_text = preprocess_text(text)
    vector = tfidf.transform([cleaned_text])

    # Logistic Regression
    lr_prediction = logistic_regression.predict(vector)[0]
    lr_probabilities = logistic_regression.predict_proba(vector)[0]
    lr_confidence = float(np.max(lr_probabilities))

    # Naive Bayes
    nb_prediction = naive_bayes.predict(vector)[0]
    nb_probabilities = naive_bayes.predict_proba(vector)[0]
    nb_confidence = float(np.max(nb_probabilities))

    # Linear SVM
    svm_prediction = linear_svm.predict(vector)[0]
    svm_score = float(
        linear_svm.decision_function(vector)[0]
    )

    # Collect predictions from all three models
    predictions = [
        int(lr_prediction),
        int(nb_prediction),
        int(svm_prediction),
    ]

    # Count votes for each class
    fake_votes = predictions.count(0)
    real_votes = predictions.count(1)

    # Determine the majority prediction
    if real_votes > fake_votes:
        voting_prediction = 1
    elif fake_votes > real_votes:
        voting_prediction = 0
    else:
        voting_prediction = None

    # Calculate agreement across the three classifiers
    if voting_prediction is not None:
        model_agreement = max(fake_votes, real_votes) / 3.0
    else:
        model_agreement = 0.0

    # Voting confidence reflects model agreement
    voting_confidence = model_agreement

    # Resolve the final ensemble result
    if voting_prediction is None:
        final_prediction = "UNCERTAIN"
    else:
        final_prediction = convert_label(voting_prediction)

    # Return the result in the existing project format
    return {
        "prediction": final_prediction,
        "confidence": round(voting_confidence, 4),
        "model_agreement": round(model_agreement, 4),
        "model_predictions": {
            "logistic_regression": convert_label(lr_prediction),
            "naive_bayes": convert_label(nb_prediction),
            "linear_svm": convert_label(svm_prediction),
        },
        "model_confidence": {
            "logistic_regression": round(lr_confidence, 4),
            "naive_bayes": round(nb_confidence, 4),
            "linear_svm": None,
        },
        "votes": {
            "FAKE": fake_votes,
            "REAL": real_votes,
        },
        "svm_decision_score": round(svm_score, 4),
        "cleaned_text": cleaned_text,
    }
