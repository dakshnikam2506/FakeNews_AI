import os
import re
import joblib
import numpy as np

from lime.lime_text import LimeTextExplainer
import shap


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
# LOAD ARTIFACTS
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


# ============================================================
# LABELS
# ============================================================

LABEL_MAP = {
    0: "FAKE",
    1: "REAL"
}

CLASS_NAMES = [
    "FAKE",
    "REAL"
]


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LIME PREDICTION FUNCTION
# ============================================================

def lime_predict(texts):

    cleaned_texts = [
        preprocess_text(text)
        for text in texts
    ]

    vectors = tfidf.transform(
        cleaned_texts
    )

    return logistic_regression.predict_proba(
        vectors
    )


# ============================================================
# LIME EXPLANATION
# ============================================================

def get_lime_explanation(
    text,
    num_features=10
):

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES
    )

    explanation = explainer.explain_instance(
        text,
        lime_predict,
        num_features=num_features
    )

    return [
        {
            "word": word,
            "weight": round(
                float(weight),
                4
            )
        }
        for word, weight
        in explanation.as_list()
    ]


# ============================================================
# SHAP EXPLAINER
# ============================================================

shap_explainer = shap.LinearExplainer(
    logistic_regression,
    tfidf.transform(
        ["sample news article"]
    )
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    text,
    num_features=10
):

    cleaned_text = preprocess_text(
        text
    )

    vector = tfidf.transform(
        [cleaned_text]
    )

    shap_result = shap_explainer(
        vector
    )

    shap_values = shap_result.values[0]

    feature_names = (
        tfidf.get_feature_names_out()
    )

    important_indices = np.argsort(
        np.abs(shap_values)
    )[::-1]

    explanation = []

    for index in important_indices:

        value = float(
            shap_values[index]
        )

        # Ignore features that are not
        # actually present in the input
        if vector[0, index] == 0:
            continue

        explanation.append(
            {
                "word": feature_names[index],
                "shap_value": round(
                    value,
                    6
                )
            }
        )

        if len(explanation) >= num_features:
            break

    return explanation


# ============================================================
# FINAL XAI FUNCTION
# ============================================================

def explain_news(
    text,
    num_features=10
):

    if not text or not str(text).strip():

        raise ValueError(
            "News text cannot be empty."
        )

    cleaned_text = preprocess_text(
        text
    )

    vector = tfidf.transform(
        [cleaned_text]
    )

    prediction = logistic_regression.predict(
        vector
    )[0]

    probabilities = (
        logistic_regression.predict_proba(
            vector
        )[0]
    )

    return {

        "model": "Logistic Regression",

        "prediction": LABEL_MAP[
            int(prediction)
        ],

        "confidence": round(
            float(
                np.max(probabilities)
            ),
            4
        ),

        "lime": get_lime_explanation(
            cleaned_text,
            num_features
        ),

        "shap": get_shap_explanation(
            cleaned_text,
            num_features
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_cases = [

        (
            "The government announced a new economic "
            "policy to increase employment and investment."
        ),

        (
            "Scientists discovered that drinking a special "
            "juice can make humans live forever."
        )
    ]

    print()
    print("=" * 70)
    print("DAY 14 — FINAL XAI INTEGRATION TEST")
    print("=" * 70)

    for number, text in enumerate(
        test_cases,
        start=1
    ):

        print()
        print("-" * 70)
        print(f"TEST {number}")
        print("-" * 70)

        print()
        print("INPUT:")
        print(text)

        result = explain_news(
            text,
            num_features=10
        )

        print()
        print(
            "PREDICTION:",
            result["prediction"]
        )

        print(
            "CONFIDENCE:",
            result["confidence"]
        )

        print()
        print("LIME:")

        for item in result["lime"]:

            print(
                f"  {item['word']:25s}"
                f"{item['weight']:+.4f}"
            )

        print()
        print("SHAP:")

        for item in result["shap"]:

            print(
                f"  {item['word']:25s}"
                f"{item['shap_value']:+.6f}"
            )

    print()
    print("=" * 70)
    print("DAY 14 XAI TEST COMPLETE")
    print("=" * 70)