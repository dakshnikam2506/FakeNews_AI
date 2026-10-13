"""Classify how retrieved evidence relates to a claim using NLI.

The model is loaded from ``models/nli-deberta-v3-small``. This module
expects the model to be present locally and does not download it at runtime.
NLI describes the relationship between two pieces of text; it does not
establish whether a claim is true in the real world.

Model labels: 0 = contradiction, 1 = entailment, 2 = neutral.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder


# Project Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nli-deberta-v3-small"
)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

_nli_model = None


# Local Model Validation

def _validate_local_model_path():
    """Check that the local model directory contains its config file.

    Raises FileNotFoundError when the directory is missing and RuntimeError
    when the path or model files are incomplete.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\n"
            "NLI LOCAL MODEL NOT FOUND\n"
            f"Expected model directory:\n{MODEL_PATH}\n\n"
            "Expected structure:\n"
            "FakeNews_AI/\n"
            "└── models/\n"
            "    └── nli-deberta-v3-small/\n\n"
            "The final pipeline will not automatically download "
            "the model."
        )

    if not MODEL_PATH.is_dir():
        raise RuntimeError(
            "\n"
            "NLI MODEL PATH IS NOT A DIRECTORY\n"
            f"Path:\n{MODEL_PATH}"
        )

    config_file = MODEL_PATH / "config.json"

    if not config_file.exists():
        raise RuntimeError(
            "\n"
            "NLI LOCAL MODEL APPEARS INCOMPLETE\n"
            f"Model directory:\n{MODEL_PATH}\n\n"
            "Missing required file:\n"
            f"{config_file}\n\n"
            "The model must be downloaded/saved completely "
            "before running the final pipeline."
        )

    return MODEL_PATH


# Model Loading

def load_nli_model():
    """Load the local CrossEncoder once and reuse it on later calls."""

    global _nli_model

    if _nli_model is not None:
        return _nli_model

    print("\n" + "=" * 70)
    print("NLI MODEL LOADING")
    print("=" * 70)

    print("\nProject root:")
    print(BASE_DIR)

    print("\nExpected local NLI model:")
    print(MODEL_PATH)

    local_model_path = _validate_local_model_path()

    print("\nLocal NLI model found.")
    print("Loading from:")
    print(local_model_path)

    try:
        _nli_model = CrossEncoder(
            str(local_model_path),
            num_labels=3,
        )

    except Exception as exc:
        _nli_model = None

        raise RuntimeError(
            "\n"
            "FAILED TO LOAD LOCAL NLI MODEL\n"
            f"Model path:\n{local_model_path}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    print("\nNLI model loaded successfully.")

    print("\nModel label mapping:")
    print("  0 -> contradiction")
    print("  1 -> entailment")
    print("  2 -> neutral")

    print("=" * 70)

    return _nli_model


# Label Normalization

def normalize_label(label):
    """Convert a model label to CONTRADICTION, ENTAILMENT, or NEUTRAL."""

    if isinstance(label, (int, np.integer)):

        mapping = {
            0: "CONTRADICTION",
            1: "ENTAILMENT",
            2: "NEUTRAL",
        }

        return mapping.get(
            int(label),
            "NEUTRAL",
        )

    label = str(label).lower().strip()

    if "contradiction" in label:
        return "CONTRADICTION"

    if "entailment" in label:
        return "ENTAILMENT"

    if "neutral" in label:
        return "NEUTRAL"

    return "NEUTRAL"


# Label Mapping

def get_label_mapping():
    """Return the label IDs used by this model."""

    return {
        0: "CONTRADICTION",
        1: "ENTAILMENT",
        2: "NEUTRAL",
    }


# Text Preparation

def prepare_evidence(
    evidence,
    max_chars=1200,
):
    """Trim evidence and cap its length before passing it to the model."""

    if not evidence:
        return ""

    evidence = str(evidence).strip()

    if len(evidence) > max_chars:
        evidence = evidence[:max_chars]

    return evidence


# Softmax

def _softmax(scores):
    """Convert the model's raw scores into probabilities."""

    scores = np.asarray(
        scores,
        dtype=float,
    )

    scores = scores - np.max(scores)

    probabilities = np.exp(scores)

    total = np.sum(probabilities)

    if total == 0:
        return np.array(
            [1 / len(scores)] * len(scores),
            dtype=float,
        )

    probabilities /= total

    return probabilities


# Single Evidence Stance

def predict_stance(
    model,
    claim,
    evidence,
):
    """Score the relationship between one evidence passage and a claim.

    The evidence is the premise and the claim is the hypothesis. The returned
    confidence is the model's score for its selected stance, not a probability
    that the claim is factually true.
    """

    evidence = prepare_evidence(
        evidence
    )

    claim = prepare_evidence(
        claim
    )

    if not claim or not evidence:
        return {
            "stance": "NEUTRAL",
            "contradiction": 0.0,
            "entailment": 0.0,
            "neutral": 1.0,
            "confidence": 1.0,
        }

    raw_scores = model.predict(
        [
            (
                evidence,
                claim,
            )
        ]
    )

    scores = np.asarray(
        raw_scores[0],
        dtype=float,
    )

    probabilities = _softmax(
        scores
    )

    contradiction_score = float(
        probabilities[0]
    )

    entailment_score = float(
        probabilities[1]
    )

    neutral_score = float(
        probabilities[2]
    )

    label_index = int(
        np.argmax(
            probabilities
        )
    )

    stance = normalize_label(
        label_index
    )

    confidence = float(
        probabilities[label_index]
    )

    return {
        "stance": stance,
        "contradiction": round(
            contradiction_score,
            6,
        ),
        "entailment": round(
            entailment_score,
            6,
        ),
        "neutral": round(
            neutral_score,
            6,
        ),
        "confidence": round(
            confidence,
            6,
        ),
    }


# Multi-Article Evidence Analysis

def analyze_evidence(
    model,
    claim,
    articles,
):
    """Run NLI on each article and summarize the observed stances.

    The aggregate describes the retrieved evidence, not a final truth label.
    """

    if not articles:
        return {
            "articles": [],
            "counts": {
                "CONTRADICTION": 0,
                "ENTAILMENT": 0,
                "NEUTRAL": 0,
            },
            "aggregate_stance": "NEUTRAL",
        }

    analyzed_articles = []

    counts = {
        "CONTRADICTION": 0,
        "ENTAILMENT": 0,
        "NEUTRAL": 0,
    }

    for article in articles:

        title = str(
            article.get(
                "title",
                "",
            )
        )

        description = str(
            article.get(
                "description",
                "",
            )
        )

        text = str(
            article.get(
                "text",
                "",
            )
        )

        evidence_parts = [
            title,
            description,
            text,
        ]

        evidence = " ".join(
            part.strip()
            for part in evidence_parts
            if part.strip()
        )

        result = predict_stance(
            model,
            claim,
            evidence,
        )

        stance = result["stance"]

        counts[stance] += 1

        enriched_article = dict(
            article
        )

        enriched_article.update(
            {
                "nli_stance": stance,
                "nli_contradiction": result[
                    "contradiction"
                ],
                "nli_entailment": result[
                    "entailment"
                ],
                "nli_neutral": result[
                    "neutral"
                ],
                "nli_confidence": result[
                    "confidence"
                ],
            }
        )

        analyzed_articles.append(
            enriched_article
        )

    # Prefer a contradiction if any article receives that label; otherwise
    # use entailment when present, and neutral when neither appears.
    if counts["CONTRADICTION"] > 0:

        aggregate_stance = (
            "CONTRADICTION"
        )

    elif counts["ENTAILMENT"] > 0:

        aggregate_stance = (
            "ENTAILMENT"
        )

    else:

        aggregate_stance = (
            "NEUTRAL"
        )

    return {
        "articles": analyzed_articles,
        "counts": counts,
        "aggregate_stance": aggregate_stance,
    }


# Standalone validation

def run_validation():

    model = load_nli_model()

    tests = [
        {
            "name": "Entailment",
            "claim": (
                "The government announced "
                "a new policy."
            ),
            "evidence": (
                "The government announced "
                "a new policy."
            ),
            "expected": "ENTAILMENT",
        },
        {
            "name": "Contradiction",
            "claim": (
                "The government cancelled "
                "the policy."
            ),
            "evidence": (
                "The government did not cancel "
                "the policy."
            ),
            "expected": "CONTRADICTION",
        },
        {
            "name": "Neutral",
            "claim": (
                "The policy will increase GDP "
                "by 10 percent."
            ),
            "evidence": (
                "The government announced "
                "the policy to encourage investment."
            ),
            "expected": "NEUTRAL",
        },
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "DAY 22 — NLI LOCAL MODEL VALIDATION"
    )

    print(
        "=" * 70
    )

    passed = 0

    for test in tests:

        result = predict_stance(
            model,
            test["claim"],
            test["evidence"],
        )

        actual = result["stance"]

        confidence = result["confidence"]

        print(
            "\n" + "-" * 70
        )

        print(
            test["name"]
        )

        print(
            "Expected:",
            test["expected"],
        )

        print(
            "Actual:",
            actual,
        )

        print(
            "Confidence:",
            confidence,
        )

        if actual == test["expected"]:

            print(
                "PASS"
            )

            passed += 1

        else:

            print(
                "REVIEW"
            )

    print(
        "\n" + "=" * 70
    )

    print(
        "DAY 22 NLI SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Passed: {passed}/{len(tests)}"
    )

    if passed == len(tests):

        print(
            "DAY 22 NLI STATUS: PASS"
        )

        return True

    print(
        "DAY 22 NLI STATUS: REVIEW"
    )

    return False


# Run validation when called directly

if __name__ == "__main__":

    run_validation()
