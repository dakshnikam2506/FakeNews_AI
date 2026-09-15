import os
import re
import joblib
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

logistic_regression = joblib.load(
    os.path.join(
        MODEL_DIR,
        "logistic_regression.pkl"
    )
)

naive_bayes = joblib.load(
    os.path.join(
        MODEL_DIR,
        "naive_bayes.pkl"
    )
)

linear_svm = joblib.load(
    os.path.join(
        MODEL_DIR,
        "linear_svm.pkl"
    )
)


# ============================================================
# LABEL MAPPING
# ============================================================

# Project label convention:
# 0 = FAKE
# 1 = REAL

LABEL_MAP = {
    0: "FAKE",
    1: "REAL"
}


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove non-alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_label(label):

    return LABEL_MAP.get(
        int(label),
        "UNKNOWN"
    )


# ============================================================
# MAIN PREDICTOR
# ============================================================

def predict_news(text):

    if not text or not str(text).strip():

        raise ValueError(
            "News text cannot be empty."
        )


    # --------------------------------------------------------
    # 1. PREPROCESS
    # --------------------------------------------------------

    cleaned_text = preprocess_text(
        text
    )


    # --------------------------------------------------------
    # 2. TF-IDF
    # --------------------------------------------------------

    vector = tfidf.transform(
        [cleaned_text]
    )


    # --------------------------------------------------------
    # 3. LOGISTIC REGRESSION
    # --------------------------------------------------------

    lr_prediction = logistic_regression.predict(
        vector
    )[0]

    lr_probabilities = logistic_regression.predict_proba(
        vector
    )[0]

    lr_confidence = float(
        np.max(lr_probabilities)
    )


    # --------------------------------------------------------
    # 4. NAIVE BAYES
    # --------------------------------------------------------

    nb_prediction = naive_bayes.predict(
        vector
    )[0]

    nb_probabilities = naive_bayes.predict_proba(
        vector
    )[0]

    nb_confidence = float(
        np.max(nb_probabilities)
    )


    # --------------------------------------------------------
    # 5. LINEAR SVM
    # --------------------------------------------------------

    svm_prediction = linear_svm.predict(
        vector
    )[0]

    svm_score = float(
        linear_svm.decision_function(
            vector
        )[0]
    )


    # --------------------------------------------------------
    # 6. MODEL PREDICTIONS
    # --------------------------------------------------------

    predictions = [
        int(lr_prediction),
        int(nb_prediction),
        int(svm_prediction)
    ]


    # --------------------------------------------------------
    # 7. MAJORITY VOTING
    # --------------------------------------------------------

    fake_votes = predictions.count(0)

    real_votes = predictions.count(1)


    if real_votes > fake_votes:

        voting_prediction = 1

    elif fake_votes > real_votes:

        voting_prediction = 0

    else:

        voting_prediction = None


    # --------------------------------------------------------
    # 8. MODEL AGREEMENT
    # --------------------------------------------------------
    #
    # Three-model ensemble:
    #
    # 3/3 models agree -> 1.0000
    # 2/3 models agree -> 0.6667
    # Tie              -> 0.0000
    #
    # This measures how strongly the three classifiers
    # agree with the majority decision.
    # --------------------------------------------------------

    if voting_prediction is not None:

        model_agreement = (
            max(
                fake_votes,
                real_votes
            ) / 3.0
        )

    else:

        model_agreement = 0.0


    # --------------------------------------------------------
    # 9. VOTING CONFIDENCE
    # --------------------------------------------------------

    # Voting confidence uses the same ensemble agreement
    # because the majority-vote strength is the confidence
    # available from this three-model voting mechanism.

    voting_confidence = model_agreement


    # --------------------------------------------------------
    # 10. FINAL RESULT
    # --------------------------------------------------------

    if voting_prediction is None:

        final_prediction = "UNCERTAIN"

    else:

        final_prediction = convert_label(
            voting_prediction
        )


    # --------------------------------------------------------
    # 11. RETURN COMPLETE PREDICTION RESULT
    # --------------------------------------------------------

    return {

        # Final ensemble prediction
        "prediction":
            final_prediction,

        # Majority-voting confidence
        "confidence":
            round(
                voting_confidence,
                4
            ),

        # Explicit model-agreement metric
        "model_agreement":
            round(
                model_agreement,
                4
            ),

        # Individual model predictions
        "model_predictions": {

            "logistic_regression":
                convert_label(
                    lr_prediction
                ),

            "naive_bayes":
                convert_label(
                    nb_prediction
                ),

            "linear_svm":
                convert_label(
                    svm_prediction
                )
        },

        # Individual model confidence
        "model_confidence": {

            "logistic_regression":
                round(
                    lr_confidence,
                    4
                ),

            "naive_bayes":
                round(
                    nb_confidence,
                    4
                ),

            # Linear SVM does not provide probability
            # unless separately calibrated.
            "linear_svm":
                None
        },

        # Vote counts
        "votes": {

            "FAKE":
                fake_votes,

            "REAL":
                real_votes
        },

        # Raw SVM decision score
        "svm_decision_score":
            round(
                svm_score,
                4
            ),

        # Preprocessed input
        "cleaned_text":
            cleaned_text
    }