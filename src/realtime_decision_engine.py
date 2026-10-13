from typing import Any, Dict, List


# Pipeline settings
TOP_K = 5
MIN_EVIDENCE_STRENGTH = 0.55
MIN_CONTRADICTION_STRENGTH = 0.50
XAI_FEATURES = 10


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, using a default if conversion fails."""

    try:
        result = float(value)
        if result != result:  # NaN
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int without raising conversion errors."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _get_domain(url: str) -> str:
    """Extract the domain name from an article URL."""

    if not url:
        return ""

    try:
        url = str(url).strip()
        url = url.replace("https://", "").replace("http://", "")

        domain = url.split("/")[0]

        if domain.startswith("www."):
            domain = domain[4:]

        return domain.lower()
    except Exception:
        return ""


def _empty_stance_summary() -> Dict[str, Any]:
    """Return the standard stance structure when no results are available."""

    return {
        "overall_stance": "INSUFFICIENT",
        "counts": {
            "SUPPORTING": 0,
            "CONTRADICTING": 0,
            "RELATED": 0,
            "INSUFFICIENT": 0,
        },
        "nli_counts": {
            "ENTAILMENT": 0,
            "CONTRADICTION": 0,
            "NEUTRAL": 0,
        },
        "average_nli_confidence": 0.0,
    }


def _empty_nli_summary() -> Dict[str, Any]:
    """Return an empty NLI summary with the expected fields."""

    return {
        "overall_stance": "INSUFFICIENT",
        "counts": {
            "SUPPORTING": 0,
            "CONTRADICTING": 0,
            "RELATED": 0,
            "INSUFFICIENT": 0,
        },
        "nli_counts": {
            "ENTAILMENT": 0,
            "CONTRADICTION": 0,
            "NEUTRAL": 0,
        },
        "average_nli_confidence": 0.0,
    }


def _normalize_xai_result(
    xai_result: Any,
) -> Dict[str, Any]:
    """Standardize XAI output so the API receives a predictable structure."""

    if not isinstance(xai_result, dict):
        return {
            "available": False,
            "model": "logistic_regression",
            "prediction": None,
            "confidence": None,
            "lime": [],
            "shap": [],
            "error": {
                "code": "XAI_INVALID_OUTPUT",
                "message": "XAI module returned an invalid output.",
            },
        }

    available = xai_result.get("available")

    if isinstance(available, bool):
        normalized_available = available
    else:
        # Older XAI implementations may not return an availability flag.
        normalized_available = bool(
            xai_result.get("lime")
            or xai_result.get("shap")
            or xai_result.get("model")
        )

    normalized = {
        "available": normalized_available,
        "model": xai_result.get(
            "model",
            "logistic_regression",
        ),
        "prediction": xai_result.get("prediction"),
        "confidence": (
            round(_safe_float(xai_result.get("confidence")), 4)
            if xai_result.get("confidence") is not None
            else None
        ),
        "lime": xai_result.get("lime", []),
        "shap": xai_result.get("shap", []),
        "error": None,
    }

    if not normalized_available:
        error_value = xai_result.get("error")

        if isinstance(error_value, dict):
            normalized["error"] = error_value
        elif error_value:
            normalized["error"] = {
                "code": "XAI_ERROR",
                "message": str(error_value),
            }
        else:
            normalized["error"] = {
                "code": "XAI_UNAVAILABLE",
                "message": "Explainability output is unavailable.",
            }

    return normalized


def _calculate_evidence_metrics(
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize the relevance and coverage of retrieved evidence.

    Evidence strength describes the available evidence. It is not
    the probability that a claim is true.
    """

    if not articles:
        return {
            "articles_evaluated": 0,
            "evidence_strength": 0.0,
            "evidence_signal": "INSUFFICIENT",
            "source_diversity": 0.0,
            "evidence_coverage": 0.0,
            "average_evidence_score": 0.0,
            "strongest_evidence_score": 0.0,
        }

    scores = []
    covered_articles = 0
    domains = set()

    for article in articles:
        if not isinstance(article, dict):
            continue

        similarity = _safe_float(article.get("similarity", 0.0))
        keyword_overlap = _safe_float(
            article.get("keyword_overlap", 0.0)
        )
        evidence_score = _safe_float(
            article.get(
                "evidence_score",
                article.get("score", 0.0),
            )
        )

        scores.append(evidence_score)

        if similarity >= 0.30 or keyword_overlap >= 0.20:
            covered_articles += 1

        domain = _get_domain(article.get("url", ""))
        if domain:
            domains.add(domain)

    if not scores:
        return {
            "articles_evaluated": 0,
            "evidence_strength": 0.0,
            "evidence_signal": "INSUFFICIENT",
            "source_diversity": 0.0,
            "evidence_coverage": 0.0,
            "average_evidence_score": 0.0,
            "strongest_evidence_score": 0.0,
        }

    strongest = max(scores)
    average = sum(scores) / len(scores)
    coverage = covered_articles / len(scores)
    source_diversity = min(len(domains) / 5.0, 1.0)

    evidence_strength = (
        0.45 * strongest
        + 0.20 * coverage
        + 0.15 * source_diversity
        + 0.20 * average
    )

    if evidence_strength >= 0.70:
        signal = "STRONG"
    elif evidence_strength >= 0.55:
        signal = "MODERATE"
    elif evidence_strength >= 0.40:
        signal = "WEAK"
    else:
        signal = "INSUFFICIENT"

    return {
        "articles_evaluated": len(scores),
        "evidence_strength": round(evidence_strength, 4),
        "evidence_signal": signal,
        "source_diversity": round(source_diversity, 4),
        "evidence_coverage": round(coverage, 4),
        "average_evidence_score": round(average, 4),
        "strongest_evidence_score": round(strongest, 4),
    }


def _build_nli_summary(
    classified_articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Count the evidence stances and calculate average NLI confidence."""

    counts = {
        "SUPPORTING": 0,
        "CONTRADICTING": 0,
        "RELATED": 0,
        "INSUFFICIENT": 0,
    }

    nli_counts = {
        "ENTAILMENT": 0,
        "CONTRADICTION": 0,
        "NEUTRAL": 0,
    }

    confidences = []

    for article in classified_articles:
        if not isinstance(article, dict):
            continue

        stance = str(
            article.get("stance", "INSUFFICIENT")
        ).upper()

        if stance in counts:
            counts[stance] += 1

        nli_stance = str(
            article.get("nli_stance", "")
        ).upper()

        if nli_stance in nli_counts:
            nli_counts[nli_stance] += 1

        confidence = article.get("nli_confidence")
        if confidence is not None:
            confidences.append(_safe_float(confidence))

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    if (
        counts["SUPPORTING"] > 0
        and counts["CONTRADICTING"] > 0
    ):
        overall_stance = "CONFLICTING"
    elif counts["SUPPORTING"] > 0:
        overall_stance = "SUPPORTING"
    elif counts["CONTRADICTING"] > 0:
        overall_stance = "CONTRADICTING"
    elif counts["RELATED"] > 0:
        overall_stance = "RELATED"
    else:
        overall_stance = "INSUFFICIENT"

    return {
        "overall_stance": overall_stance,
        "counts": counts,
        "nli_counts": nli_counts,
        "average_nli_confidence": round(average_confidence, 4),
    }


def make_realtime_decision(
    ml_result: Dict[str, Any],
    evidence_result: Dict[str, Any],
    evidence_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the ML prediction with the retrieved evidence and its stance."""

    if not isinstance(ml_result, dict):
        ml_result = {}

    if not isinstance(evidence_result, dict):
        evidence_result = {}

    if not isinstance(evidence_analysis, dict):
        evidence_analysis = {}

    ml_prediction = str(
        ml_result.get("prediction", "")
    ).upper().strip()

    if ml_prediction not in {"REAL", "FAKE"}:
        return {
            "decision": "UNCERTAIN",
            "reason": "Invalid or unavailable ML prediction.",
            "ml_prediction": ml_prediction,
            "evidence_strength": 0.0,
            "evidence_signal": "INSUFFICIENT",
            "evidence_stance": "INSUFFICIENT",
            "articles_evaluated": 0,
            "supporting_count": 0,
            "contradicting_count": 0,
            "related_count": 0,
            "insufficient_count": 0,
        }

    evidence_strength = _safe_float(
        evidence_result.get(
            "evidence_strength",
            evidence_analysis.get("evidence_strength", 0.0),
        )
    )

    evidence_signal = str(
        evidence_result.get(
            "evidence_signal",
            evidence_analysis.get(
                "evidence_signal",
                "INSUFFICIENT",
            ),
        )
    ).upper()

    articles_evaluated = _safe_int(
        evidence_result.get(
            "articles_evaluated",
            evidence_analysis.get("articles_evaluated", 0),
        )
    )

    stance_summary = evidence_analysis.get("stance_summary", {})
    if not isinstance(stance_summary, dict):
        stance_summary = {}

    evidence_stance = str(
        stance_summary.get(
            "overall_stance",
            evidence_analysis.get(
                "overall_stance",
                "INSUFFICIENT",
            ),
        )
    ).upper()

    counts = stance_summary.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    supporting_count = _safe_int(counts.get("SUPPORTING", 0))
    contradicting_count = _safe_int(counts.get("CONTRADICTING", 0))
    related_count = _safe_int(counts.get("RELATED", 0))
    insufficient_count = _safe_int(counts.get("INSUFFICIENT", 0))

    base_result = {
        "ml_prediction": ml_prediction,
        "evidence_strength": round(evidence_strength, 4),
        "evidence_signal": evidence_signal,
        "evidence_stance": evidence_stance,
        "articles_evaluated": articles_evaluated,
        "supporting_count": supporting_count,
        "contradicting_count": contradicting_count,
        "related_count": related_count,
        "insufficient_count": insufficient_count,
    }

    # Weak evidence should not normally produce a definitive result.
    if evidence_strength < MIN_EVIDENCE_STRENGTH:
        if (
            evidence_stance == "CONTRADICTING"
            and evidence_strength >= MIN_CONTRADICTION_STRENGTH
            and contradicting_count >= 1
        ):
            decision = "FAKE"
            reason = (
                "The retrieved evidence contradicts the claim "
                "with sufficient evidence strength."
            )
        else:
            decision = "UNCERTAIN"
            reason = (
                "The available evidence is not strong enough "
                "to make a reliable hybrid decision."
            )

        return {
            "decision": decision,
            "reason": reason,
            **base_result,
        }

    # Mixed supporting and contradicting evidence needs further review.
    if evidence_stance == "CONFLICTING":
        return {
            "decision": "UNCERTAIN",
            "reason": (
                "Retrieved evidence contains both supporting "
                "and contradicting signals."
            ),
            **base_result,
        }

    # Related articles alone are not enough to verify the claim.
    if evidence_stance in {"RELATED", "INSUFFICIENT"}:
        return {
            "decision": "UNCERTAIN",
            "reason": (
                "The retrieved evidence is related to the claim "
                "but does not sufficiently support or contradict it."
            ),
            **base_result,
        }

    # Supporting evidence must agree with the ML classification.
    if evidence_stance == "SUPPORTING":
        if ml_prediction == "REAL":
            decision = "REAL"
            reason = (
                "The ML classifier predicts REAL and the "
                "retrieved real-time evidence supports the claim."
            )
        else:
            decision = "UNCERTAIN"
            reason = (
                "The retrieved evidence supports the claim, "
                "but the ML classifier predicts FAKE."
            )

        return {
            "decision": decision,
            "reason": reason,
            **base_result,
        }

    # Contradicting evidence can support a FAKE result when strong enough.
    if evidence_stance == "CONTRADICTING":
        if (
            evidence_strength >= MIN_CONTRADICTION_STRENGTH
            and contradicting_count >= 1
        ):
            decision = "FAKE"
            reason = (
                "The retrieved evidence contradicts the claim "
                "with sufficient evidence strength."
            )
        else:
            decision = "UNCERTAIN"
            reason = (
                "Contradicting evidence was detected, but its "
                "strength is insufficient for a reliable FAKE decision."
            )

        return {
            "decision": decision,
            "reason": reason,
            **base_result,
        }

    return {
        "decision": "UNCERTAIN",
        "reason": (
            "The evidence pattern does not satisfy a reliable "
            "hybrid decision rule."
        ),
        **base_result,
    }


def run_realtime_pipeline(
    claim: str,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Run classification, evidence analysis, decision-making, and XAI."""

    from .predictor import predict_news

    from .realtime_semantic_retriever import (
        retrieve_realtime_semantic_evidence,
    )

    from .realtime_verifier import (
        verify_articles,
        summarize_evidence,
    )

    from .evidence_stance import (
        classify_evidence_articles,
        summarize_stances,
    )

    from .xai import explain_news

    # Handle empty input before calling any models.
    if not claim or not str(claim).strip():
        empty_stance = _empty_stance_summary()

        return {
            "claim": claim,
            "ml_prediction": "UNKNOWN",
            "model_agreement": 0.0,
            "ml_result": {},
            "evidence_strength": 0.0,
            "evidence_signal": "INSUFFICIENT",
            "articles_evaluated": 0,
            "source_diversity": 0.0,
            "evidence_coverage": 0.0,
            "average_evidence_score": 0.0,
            "strongest_evidence_score": 0.0,
            "evidence_stance": "INSUFFICIENT",
            "stance_summary": empty_stance,
            "articles": [],
            "nli_summary": _empty_nli_summary(),
            "hybrid_decision": "UNCERTAIN",
            "decision": "UNCERTAIN",
            "reason": "Empty claim.",
            "xai": {
                "available": False,
                "model": "logistic_regression",
                "prediction": None,
                "confidence": None,
                "lime": [],
                "shap": [],
                "error": {
                    "code": "EMPTY_CLAIM",
                    "message": "Cannot generate XAI for an empty claim.",
                },
            },
        }

    claim = str(claim).strip()

    print()
    print("=" * 70)
    print("HYBRID REAL-TIME FAKE NEWS DETECTION")
    print("=" * 70)
    print("\nCLAIM:")
    print(claim)

    # Step 1: ML classification
    print()
    print("-" * 70)
    print("STEP 1 - ML CLASSIFICATION")
    print("-" * 70)

    try:
        ml_result = predict_news(claim)
        if not isinstance(ml_result, dict):
            ml_result = {}
    except Exception as exc:
        ml_result = {
            "prediction": "UNKNOWN",
            "model_agreement": 0.0,
            "error": {
                "code": "ML_ERROR",
                "message": str(exc),
            },
        }

    ml_prediction = str(
        ml_result.get("prediction", "UNKNOWN")
    ).upper()

    model_agreement = _safe_float(
        ml_result.get("model_agreement", 0.0)
    )

    print(f"ML Prediction: {ml_prediction}")
    print(f"Model Agreement: {model_agreement:.4f}")

    # Step 2: Retrieve relevant articles
    print()
    print("-" * 70)
    print("STEP 2 - REAL-TIME EVIDENCE RETRIEVAL")
    print("-" * 70)

    retrieval_error = None

    try:
        articles = retrieve_realtime_semantic_evidence(
            claim,
            top_k=top_k,
        )
    except Exception as exc:
        articles = []
        retrieval_error = {
            "code": "RETRIEVAL_ERROR",
            "message": str(exc),
        }

    if not isinstance(articles, list):
        articles = []

    articles = [
        article
        for article in articles
        if isinstance(article, dict)
    ]

    print(f"Retrieved Articles: {len(articles)}")

    # Step 3: Verify the retrieved articles
    print()
    print("-" * 70)
    print("STEP 3 - EVIDENCE VERIFICATION")
    print("-" * 70)

    verification_error = None

    try:
        verified_articles = verify_articles(claim, articles)
    except Exception as exc:
        verified_articles = []
        verification_error = {
            "code": "VERIFICATION_ERROR",
            "message": str(exc),
        }

    if not isinstance(verified_articles, list):
        verified_articles = []

    verified_articles = [
        article
        for article in verified_articles
        if isinstance(article, dict)
    ]

    try:
        evidence_result = summarize_evidence(claim, articles)
        if not isinstance(evidence_result, dict):
            evidence_result = {}
    except Exception as exc:
        evidence_result = {}
        verification_error = {
            "code": "EVIDENCE_SUMMARY_ERROR",
            "message": str(exc),
        }

    # Step 4: Classify the evidence stance
    print()
    print("-" * 70)
    print("STEP 4 - EVIDENCE STANCE / NLI")
    print("-" * 70)

    stance_error = None

    try:
        classified_articles = classify_evidence_articles(
            claim,
            verified_articles,
        )
    except Exception as exc:
        classified_articles = []
        stance_error = {
            "code": "STANCE_ERROR",
            "message": str(exc),
        }

    if not isinstance(classified_articles, list):
        classified_articles = []

    classified_articles = [
        article
        for article in classified_articles
        if isinstance(article, dict)
    ]

    try:
        stance_summary = summarize_stances(classified_articles)
        if not isinstance(stance_summary, dict):
            stance_summary = _empty_stance_summary()
    except Exception as exc:
        stance_summary = _empty_stance_summary()
        stance_error = {
            "code": "STANCE_SUMMARY_ERROR",
            "message": str(exc),
        }

    # Step 5: Calculate evidence metrics
    metrics = _calculate_evidence_metrics(verified_articles)

    # Use the verifier's aggregate values when it provides them.
    for key in [
        "evidence_strength",
        "evidence_signal",
        "articles_evaluated",
        "source_diversity",
        "evidence_coverage",
        "average_evidence_score",
        "strongest_evidence_score",
    ]:
        if key not in evidence_result:
            continue

        value = evidence_result[key]

        if key == "evidence_signal":
            metrics[key] = str(value).upper()
        elif key == "articles_evaluated":
            metrics[key] = _safe_int(value)
        else:
            metrics[key] = round(_safe_float(value), 4)

    print("\nEvidence Signal:")
    print(metrics["evidence_signal"])
    print("Evidence Strength:")
    print(f"{metrics['evidence_strength']:.4f}")
    print("Articles Evaluated:")
    print(metrics["articles_evaluated"])
    print("\nEvidence Stance:")
    print(stance_summary)

    # Step 6: Combine the ML and evidence results
    evidence_analysis = {
        "evidence_strength": metrics["evidence_strength"],
        "evidence_signal": metrics["evidence_signal"],
        "articles_evaluated": metrics["articles_evaluated"],
        "stance_summary": stance_summary,
    }

    decision_result = make_realtime_decision(
        ml_result,
        evidence_result,
        evidence_analysis,
    )

    final_decision = str(
        decision_result.get("decision", "UNCERTAIN")
    ).upper()

    # Step 7: Generate LIME and SHAP explanations
    print()
    print("-" * 70)
    print("STEP 7 - EXPLAINABLE AI")
    print("-" * 70)

    try:
        # explain_news expects num_features, not max_features.
        xai_result = explain_news(
            claim,
            num_features=XAI_FEATURES,
        )
        xai_result = _normalize_xai_result(xai_result)
    except Exception as exc:
        xai_result = {
            "available": False,
            "model": "logistic_regression",
            "prediction": None,
            "confidence": None,
            "lime": [],
            "shap": [],
            "error": {
                "code": "XAI_ERROR",
                "message": str(exc),
            },
        }

    print(f"XAI available: {xai_result['available']}")

    # Print the final result
    print()
    print("=" * 70)
    print("FINAL HYBRID DECISION")
    print("=" * 70)
    print(f"\nML Prediction: {ml_prediction}")
    print(f"Evidence Signal: {metrics['evidence_signal']}")
    print(f"Evidence Strength: {metrics['evidence_strength']:.4f}")
    print(
        "Evidence Stance: "
        f"{stance_summary.get('overall_stance', 'INSUFFICIENT')}"
    )
    print(f"\nFINAL DECISION: {final_decision}")
    print(f"\nREASON: {decision_result.get('reason', '')}")
    print("=" * 70)

    # Collect any errors encountered during the pipeline.
    errors = []

    if retrieval_error:
        errors.append(retrieval_error)

    if verification_error:
        errors.append(verification_error)

    if stance_error:
        errors.append(stance_error)

    if isinstance(ml_result.get("error"), dict):
        errors.append(ml_result["error"])

    if not xai_result.get("available", False):
        xai_error = xai_result.get("error")
        if isinstance(xai_error, dict):
            errors.append(xai_error)

    error_output = errors if errors else None

    # Keep the result structure stable for the API.
    return {
        "claim": claim,
        "ml_prediction": ml_prediction,
        "model_agreement": round(model_agreement, 4),
        "ml_result": ml_result,
        "evidence_strength": metrics["evidence_strength"],
        "evidence_signal": metrics["evidence_signal"],
        "articles_evaluated": metrics["articles_evaluated"],
        "source_diversity": metrics["source_diversity"],
        "evidence_coverage": metrics["evidence_coverage"],
        "average_evidence_score": metrics["average_evidence_score"],
        "strongest_evidence_score": metrics["strongest_evidence_score"],
        "evidence_stance": stance_summary.get(
            "overall_stance",
            "INSUFFICIENT",
        ),
        "stance_summary": stance_summary,
        "articles": classified_articles,
        "nli_summary": _build_nli_summary(classified_articles),
        "decision": final_decision,
        "reason": decision_result.get("reason", ""),
        "xai": xai_result,
        "errors": error_output,
    }


if __name__ == "__main__":
    test_claim = "The government announced a new industrial policy."

    result = run_realtime_pipeline(test_claim)

    print("\nPipeline completed.")
    print(
        "Final decision:",
        result.get("decision", "UNCERTAIN"),
    )
    print(
        "XAI available:",
        result.get("xai", {}).get("available", False),
    )
