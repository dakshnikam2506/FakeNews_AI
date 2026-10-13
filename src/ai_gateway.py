```python
"""
FakeNews_AI - AI Gateway

This file connects the backend to the AI pipeline.
The backend can use predict_claim() to analyze a news claim
without handling the individual AI components directly.
"""

from typing import Any, Dict
import json

from .realtime_decision_engine import run_realtime_pipeline


VALID_DECISIONS = {
    "REAL",
    "FAKE",
    "UNCERTAIN",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _normalize_decision(value: Any) -> str:
    value = str(value or "UNCERTAIN").upper()

    if value in VALID_DECISIONS:
        return value

    return "UNCERTAIN"


def _extract_nli_counts(
    nli_raw: Dict[str, Any],
    articles: list,
) -> Dict[str, int]:
    # Read the NLI summary first, then use article results if needed.
    counts = _safe_dict(nli_raw.get("counts"))

    entailment = _safe_int(
        counts.get("entailment"),
        _safe_int(nli_raw.get("entailment"), 0),
    )

    contradiction = _safe_int(
        counts.get("contradiction"),
        _safe_int(nli_raw.get("contradiction"), 0),
    )

    neutral = _safe_int(
        counts.get("neutral"),
        _safe_int(nli_raw.get("neutral"), 0),
    )

    if (
        entailment == 0
        and contradiction == 0
        and neutral == 0
        and articles
    ):
        for article in articles:
            if not isinstance(article, dict):
                continue

            stance = str(article.get("nli_stance", "")).upper()

            if stance == "ENTAILMENT":
                entailment += 1
            elif stance == "CONTRADICTION":
                contradiction += 1
            elif stance == "NEUTRAL":
                neutral += 1

    return {
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
    }


def _normalize_xai(
    xai_raw: Dict[str, Any],
    ml_prediction: str,
) -> Dict[str, Any]:
    available = bool(xai_raw.get("available", False))

    lime = xai_raw.get("lime", [])
    shap = xai_raw.get("shap", [])

    if not isinstance(lime, list):
        lime = []

    if not isinstance(shap, list):
        shap = []

    return {
        "available": available,
        "model": xai_raw.get(
            "model",
            "Logistic Regression",
        ),
        "prediction": xai_raw.get(
            "prediction",
            ml_prediction,
        ),
        "confidence": _safe_float(
            xai_raw.get("confidence"),
            0.0,
        ),
        "lime": lime,
        "shap": shap,
        "error": xai_raw.get("error"),
    }


def _normalize_errors(
    raw_result: Dict[str, Any],
) -> Any:
    errors = raw_result.get("errors", [])

    if errors is None:
        return None

    if isinstance(errors, list):
        return errors if errors else None

    return errors


def _build_public_response(
    raw_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare the pipeline result for the backend API."""

    raw_result = _safe_dict(raw_result)

    claim = str(raw_result.get("claim", ""))

    decision = _normalize_decision(
        raw_result.get("decision")
        or raw_result.get("hybrid_decision")
    )

    reason = str(
        raw_result.get("reason")
        or "The system could not establish sufficient evidence."
    )

    articles = raw_result.get("articles", [])

    if not isinstance(articles, list):
        articles = []

    ml_result = _safe_dict(
        raw_result.get("ml_result")
        or raw_result.get("ml")
    )

    ml_prediction = str(
        ml_result.get("prediction", "UNCERTAIN")
    ).upper()

    model_predictions = _safe_dict(
        ml_result.get("model_predictions")
    )

    ml_confidence = _safe_float(
        ml_result.get("confidence"),
        0.0,
    )

    model_agreement = _safe_float(
        ml_result.get("model_agreement"),
        0.0,
    )

    ml = {
        "prediction": ml_prediction,
        "confidence": ml_confidence,
        "model_agreement": model_agreement,
        "model_predictions": model_predictions,
    }

    evidence_signal = str(
        raw_result.get(
            "evidence_signal",
            raw_result.get("signal", "INSUFFICIENT"),
        )
    ).upper()

    evidence_strength = _safe_float(
        raw_result.get("evidence_strength", 0.0),
        0.0,
    )

    stance_summary = _safe_dict(
        raw_result.get("stance_summary")
    )

    evidence_stance = str(
        stance_summary.get(
            "overall_stance",
            raw_result.get("evidence_stance", "INSUFFICIENT"),
        )
    ).upper()

    articles_evaluated = _safe_int(
        raw_result.get("articles_evaluated"),
        len(articles),
    )

    evidence = {
        "signal": evidence_signal,
        "strength": evidence_strength,
        "articles_evaluated": articles_evaluated,
        "stance": evidence_stance,
    }

    nli_raw = _safe_dict(
        raw_result.get("nli_summary")
    )

    nli_counts = _extract_nli_counts(
        nli_raw,
        articles,
    )

    nli = {
        "entailment": nli_counts["entailment"],
        "contradiction": nli_counts["contradiction"],
        "neutral": nli_counts["neutral"],
    }

    xai_raw = _safe_dict(raw_result.get("xai"))

    xai = _normalize_xai(
        xai_raw,
        ml_prediction,
    )

    error = _normalize_errors(raw_result)

    return {
        "claim": claim,
        "decision": decision,
        "reason": reason,
        "ml": ml,
        "evidence": evidence,
        "nli": nli,
        "articles": articles,
        "xai": xai,
        "error": error,
    }


def predict_claim(
    claim: str,
) -> Dict[str, Any]:
    """Run the AI pipeline for a news claim and return its result."""

    if not isinstance(claim, str):
        return {
            "claim": "",
            "decision": "UNCERTAIN",
            "reason": "Claim must be a string.",
            "ml": {},
            "evidence": {},
            "nli": {},
            "articles": [],
            "xai": {
                "available": False,
                "model": "Logistic Regression",
                "prediction": None,
                "confidence": 0.0,
                "lime": [],
                "shap": [],
                "error": "INVALID_CLAIM_TYPE",
            },
            "error": "INVALID_CLAIM_TYPE",
        }

    claim = claim.strip()

    if not claim:
        return {
            "claim": "",
            "decision": "UNCERTAIN",
            "reason": "Claim cannot be empty.",
            "ml": {},
            "evidence": {},
            "nli": {},
            "articles": [],
            "xai": {
                "available": False,
                "model": "Logistic Regression",
                "prediction": None,
                "confidence": 0.0,
                "lime": [],
                "shap": [],
                "error": "EMPTY_CLAIM",
            },
            "error": "EMPTY_CLAIM",
        }

    try:
        raw_result = run_realtime_pipeline(claim)
        return _build_public_response(raw_result)

    except Exception as exc:
        return {
            "claim": claim,
            "decision": "UNCERTAIN",
            "reason": (
                "The AI pipeline encountered "
                "an internal error."
            ),
            "ml": {},
            "evidence": {},
            "nli": {},
            "articles": [],
            "xai": {
                "available": False,
                "model": "Logistic Regression",
                "prediction": None,
                "confidence": 0.0,
                "lime": [],
                "shap": [],
                "error": "AI_PIPELINE_ERROR",
            },
            "error": {
                "code": "AI_PIPELINE_ERROR",
                "message": str(exc),
            },
        }


if __name__ == "__main__":
    test_claim = (
        "The government announced "
        "a new industrial policy."
    )

    print("=" * 70)
    print("FAKENEWS_AI - AI GATEWAY TEST")
    print("=" * 70)

    print("\nClaim:")
    print(test_claim)

    print("\nRunning AI pipeline...")

    result = predict_claim(test_claim)

    print("\n" + "=" * 70)
    print("FINAL PUBLIC API RESPONSE")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n" + "=" * 70)
    print("GATEWAY TEST COMPLETE")
    print("=" * 70)

    print("\nFinal decision:", result.get("decision"))
    print(
        "XAI available:",
        result.get("xai", {}).get("available", False),
    )
    print("NLI:", result.get("nli"))
```
