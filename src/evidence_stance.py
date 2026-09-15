"""
Evidence Stance Classification
==============================

Classifies retrieved evidence relative to a claim as:

    SUPPORTING
    CONTRADICTING
    RELATED
    INSUFFICIENT

The classifier uses:

    1. Semantic similarity
    2. Keyword overlap
    3. Contradiction heuristics
    4. NLI using DeBERTa

Important:
    This module is an evidence stance classifier.
    It is NOT a truth detector.

NLI determines the textual relationship between
the evidence and the claim.

NLI input order:

    Evidence = premise
    Claim    = hypothesis
"""

import re
from typing import Dict, List


# ============================================================
# NLI IMPORT
# ============================================================

try:
    from .nli_stance import load_nli_model, predict_stance
    NLI_AVAILABLE = True

except ImportError:
    try:
        from nli_stance import load_nli_model, predict_stance
        NLI_AVAILABLE = True

    except ImportError:
        NLI_AVAILABLE = False
        load_nli_model = None
        predict_stance = None


# ============================================================
# CONFIGURATION
# ============================================================

MIN_SIMILARITY = 0.30
STRONG_SIMILARITY = 0.70

MIN_KEYWORD_OVERLAP = 0.20
STRONG_KEYWORD_OVERLAP = 0.50

# NLI is only trusted when the evidence is
# sufficiently relevant to the claim.
NLI_MIN_SIMILARITY = 0.40
NLI_MIN_KEYWORD_OVERLAP = 0.20
NLI_MIN_CONFIDENCE = 0.70


# ============================================================
# GLOBAL NLI MODEL
# ============================================================

_nli_model = None


def get_nli_model():
    """
    Load the NLI model lazily.

    The model is loaded only when NLI is required.
    """

    global _nli_model

    if not NLI_AVAILABLE:
        return None

    if _nli_model is None:
        _nli_model = load_nli_model()

    return _nli_model


# ============================================================
# NEGATION / CONTRADICTION WORDS
# ============================================================

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "false",
    "falsely",
    "denied",
    "deny",
    "denies",
    "reject",
    "rejects",
    "rejected",
    "cancelled",
    "canceled",
    "withdrawn",
    "withdraw",
    "scrapped",
    "scrap",
    "halted",
    "halt",
    "blocked",
    "block",
    "refused",
    "refuse",
    "debunked",
    "debunk",
}


# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize text for keyword analysis.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Keep only letters, numbers and spaces.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def tokenize(text: str) -> List[str]:
    """
    Convert normalized text into word tokens.
    """

    return clean_text(text).split()


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "has",
    "have",
    "had",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "into",
    "about",
    "after",
    "before",
    "today",
    "yesterday",
}


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text: str) -> set:
    """
    Extract informative words from text.
    """

    tokens = tokenize(text)

    return {
        token
        for token in tokens
        if len(token) >= 3
        and token not in STOPWORDS
    }


def keyword_overlap(
    claim: str,
    evidence: str,
) -> float:
    """
    Calculate the percentage of claim keywords
    appearing in the evidence.

    Returns:

        0.0 = no overlap
        1.0 = complete overlap
    """

    claim_keywords = extract_keywords(claim)
    evidence_keywords = extract_keywords(evidence)

    if not claim_keywords:
        return 0.0

    overlap = claim_keywords.intersection(
        evidence_keywords
    )

    return len(overlap) / len(claim_keywords)


# ============================================================
# CONTRADICTION HEURISTIC
# ============================================================

def contains_negation(text: str) -> bool:
    """
    Check whether text contains explicit
    negation or contradiction vocabulary.
    """

    tokens = tokenize(text)

    return any(
        token in NEGATION_WORDS
        for token in tokens
    )


def contradiction_signal(
    claim: str,
    evidence: str,
) -> float:
    """
    Calculate a simple contradiction-language signal.

    This is only a supporting heuristic.
    NLI remains the primary stance mechanism.
    """

    evidence_tokens = tokenize(evidence)

    if not evidence_tokens:
        return 0.0

    negation_count = sum(
        1
        for token in evidence_tokens
        if token in NEGATION_WORDS
    )

    if negation_count == 0:
        return 0.0

    return min(
        1.0,
        negation_count / 3.0,
    )


# ============================================================
# NLI
# ============================================================

def run_nli(
    claim: str,
    evidence: str,
) -> Dict:
    """
    Run NLI for one claim/evidence pair.

    The underlying NLI function expects:

        evidence -> premise
        claim    -> hypothesis

    Important:
        nli_stance.py returns keys named:

            contradiction
            entailment
            neutral
            confidence

        This function normalizes them into the
        names used by evidence_stance.py.
    """

    unavailable_result = {
        "available": False,
        "stance": "UNAVAILABLE",
        "contradiction_score": 0.0,
        "entailment_score": 0.0,
        "neutral_score": 0.0,
        "confidence": 0.0,
    }

    if not NLI_AVAILABLE:
        return unavailable_result

    try:
        model = get_nli_model()

        if model is None:
            return unavailable_result

        result = predict_stance(
            model,
            claim,
            evidence,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Support both the current nli_stance.py format and
        # the older *_score format.
        # ----------------------------------------------------

        contradiction_score = result.get(
            "contradiction",
            result.get(
                "contradiction_score",
                0.0,
            ),
        )

        entailment_score = result.get(
            "entailment",
            result.get(
                "entailment_score",
                0.0,
            ),
        )

        neutral_score = result.get(
            "neutral",
            result.get(
                "neutral_score",
                0.0,
            ),
        )

        confidence = result.get(
            "confidence",
            max(
                float(contradiction_score),
                float(entailment_score),
                float(neutral_score),
            ),
        )

        return {
            "available": True,

            "stance": str(
                result.get(
                    "stance",
                    "UNKNOWN",
                )
            ).upper(),

            "contradiction_score": float(
                contradiction_score
            ),

            "entailment_score": float(
                entailment_score
            ),

            "neutral_score": float(
                neutral_score
            ),

            "confidence": float(
                confidence
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "stance": "ERROR",
            "contradiction_score": 0.0,
            "entailment_score": 0.0,
            "neutral_score": 0.0,
            "confidence": 0.0,
            "error": str(exc),
        }


# ============================================================
# STANCE CLASSIFICATION
# ============================================================

def classify_stance(
    claim: str,
    evidence: str,
    similarity: float = 0.0,
) -> Dict:
    """
    Classify evidence relative to a claim.

    Possible results:

        SUPPORTING
        CONTRADICTING
        RELATED
        INSUFFICIENT

    Decision process:

        1. Check input.
        2. Calculate keyword overlap.
        3. Apply relevance gate.
        4. Run NLI.
        5. Respect high-confidence NLI contradiction.
        6. Respect high-confidence NLI entailment.
        7. Treat NLI neutral as RELATED.
        8. Use conservative fallback if NLI fails.

    This function does NOT determine factual truth.
    """

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not claim or not evidence:
        return {
            "stance": "INSUFFICIENT",
            "similarity": float(similarity),
            "keyword_overlap": 0.0,
            "contradiction_signal": 0.0,
            "nli_stance": "UNAVAILABLE",
            "nli_contradiction": 0.0,
            "nli_entailment": 0.0,
            "nli_neutral": 0.0,
            "nli_confidence": 0.0,
            "reason": "Claim or evidence text is empty.",
        }

    similarity = float(similarity)

    # --------------------------------------------------------
    # HEURISTIC SIGNALS
    # --------------------------------------------------------

    overlap = keyword_overlap(
        claim,
        evidence,
    )

    contradiction = contradiction_signal(
        claim,
        evidence,
    )

    # --------------------------------------------------------
    # BASIC RELEVANCE GATE
    # --------------------------------------------------------

    if (
        similarity < MIN_SIMILARITY
        and overlap < MIN_KEYWORD_OVERLAP
    ):
        return {
            "stance": "INSUFFICIENT",
            "similarity": round(similarity, 4),
            "keyword_overlap": round(overlap, 4),
            "contradiction_signal": round(
                contradiction,
                4,
            ),
            "nli_stance": "SKIPPED",
            "nli_contradiction": 0.0,
            "nli_entailment": 0.0,
            "nli_neutral": 0.0,
            "nli_confidence": 0.0,
            "reason": (
                "Evidence is not sufficiently "
                "relevant to the claim."
            ),
        }

    # --------------------------------------------------------
    # NLI RELEVANCE GATE
    # --------------------------------------------------------

    sufficient_relevance = (
        similarity >= NLI_MIN_SIMILARITY
        and overlap >= NLI_MIN_KEYWORD_OVERLAP
    )

    # --------------------------------------------------------
    # RUN NLI
    # --------------------------------------------------------

    nli_result = run_nli(
        claim,
        evidence,
    )

    nli_stance = nli_result.get(
        "stance",
        "UNKNOWN",
    )

    nli_contradiction = float(
        nli_result.get(
            "contradiction_score",
            0.0,
        )
    )

    nli_entailment = float(
        nli_result.get(
            "entailment_score",
            0.0,
        )
    )

    nli_neutral = float(
        nli_result.get(
            "neutral_score",
            0.0,
        )
    )

    nli_confidence = float(
        nli_result.get(
            "confidence",
            max(
                nli_contradiction,
                nli_entailment,
                nli_neutral,
            ),
        )
    )

    # --------------------------------------------------------
    # NLI CONTRADICTION
    #
    # IMPORTANT:
    # This check comes before entailment and fallback logic.
    # --------------------------------------------------------

    if (
        sufficient_relevance
        and nli_stance == "CONTRADICTION"
        and nli_contradiction >= NLI_MIN_CONFIDENCE
    ):
        return {
            "stance": "CONTRADICTING",
            "similarity": round(similarity, 4),
            "keyword_overlap": round(overlap, 4),
            "contradiction_signal": round(
                contradiction,
                4,
            ),
            "nli_stance": nli_stance,
            "nli_contradiction": round(
                nli_contradiction,
                4,
            ),
            "nli_entailment": round(
                nli_entailment,
                4,
            ),
            "nli_neutral": round(
                nli_neutral,
                4,
            ),
            "nli_confidence": round(
                nli_confidence,
                4,
            ),
            "reason": (
                "Relevant evidence is classified "
                "by NLI as contradicting the claim."
            ),
        }

    # --------------------------------------------------------
    # NLI ENTAILMENT
    # --------------------------------------------------------

    if (
        sufficient_relevance
        and nli_stance == "ENTAILMENT"
        and nli_entailment >= NLI_MIN_CONFIDENCE
    ):
        return {
            "stance": "SUPPORTING",
            "similarity": round(similarity, 4),
            "keyword_overlap": round(overlap, 4),
            "contradiction_signal": round(
                contradiction,
                4,
            ),
            "nli_stance": nli_stance,
            "nli_contradiction": round(
                nli_contradiction,
                4,
            ),
            "nli_entailment": round(
                nli_entailment,
                4,
            ),
            "nli_neutral": round(
                nli_neutral,
                4,
            ),
            "nli_confidence": round(
                nli_confidence,
                4,
            ),
            "reason": (
                "Relevant evidence is classified "
                "by NLI as supporting the claim."
            ),
        }

    # --------------------------------------------------------
    # NLI NEUTRAL
    # --------------------------------------------------------

    if (
        sufficient_relevance
        and nli_stance == "NEUTRAL"
    ):
        return {
            "stance": "RELATED",
            "similarity": round(similarity, 4),
            "keyword_overlap": round(overlap, 4),
            "contradiction_signal": round(
                contradiction,
                4,
            ),
            "nli_stance": nli_stance,
            "nli_contradiction": round(
                nli_contradiction,
                4,
            ),
            "nli_entailment": round(
                nli_entailment,
                4,
            ),
            "nli_neutral": round(
                nli_neutral,
                4,
            ),
            "nli_confidence": round(
                nli_confidence,
                4,
            ),
            "reason": (
                "Evidence is relevant, but NLI "
                "does not establish support or "
                "contradiction."
            ),
        }

    # --------------------------------------------------------
    # NLI ERROR / UNAVAILABLE
    # --------------------------------------------------------

    if nli_stance in {
        "ERROR",
        "UNAVAILABLE",
    }:
        # Conservative fallback.
        #
        # Only strong semantic and lexical
        # alignment is considered supporting.

        if (
            similarity >= STRONG_SIMILARITY
            and overlap >= STRONG_KEYWORD_OVERLAP
            and contradiction == 0.0
        ):
            return {
                "stance": "SUPPORTING",
                "similarity": round(
                    similarity,
                    4,
                ),
                "keyword_overlap": round(
                    overlap,
                    4,
                ),
                "contradiction_signal": round(
                    contradiction,
                    4,
                ),
                "nli_stance": nli_stance,
                "nli_contradiction": round(
                    nli_contradiction,
                    4,
                ),
                "nli_entailment": round(
                    nli_entailment,
                    4,
                ),
                "nli_neutral": round(
                    nli_neutral,
                    4,
                ),
                "nli_confidence": round(
                    nli_confidence,
                    4,
                ),
                "reason": (
                    "NLI unavailable; evidence is "
                    "strongly semantically and "
                    "lexically aligned."
                ),
            }

    # --------------------------------------------------------
    # RELATED
    # --------------------------------------------------------

    if (
        similarity >= MIN_SIMILARITY
        or overlap >= MIN_KEYWORD_OVERLAP
    ):
        return {
            "stance": "RELATED",
            "similarity": round(
                similarity,
                4,
            ),
            "keyword_overlap": round(
                overlap,
                4,
            ),
            "contradiction_signal": round(
                contradiction,
                4,
            ),
            "nli_stance": nli_stance,
            "nli_contradiction": round(
                nli_contradiction,
                4,
            ),
            "nli_entailment": round(
                nli_entailment,
                4,
            ),
            "nli_neutral": round(
                nli_neutral,
                4,
            ),
            "nli_confidence": round(
                nli_confidence,
                4,
            ),
            "reason": (
                "Evidence is related to the claim "
                "but does not provide sufficiently "
                "strong support or contradiction."
            ),
        }

    # --------------------------------------------------------
    # FINAL FALLBACK
    # --------------------------------------------------------

    return {
        "stance": "INSUFFICIENT",
        "similarity": round(
            similarity,
            4,
        ),
        "keyword_overlap": round(
            overlap,
            4,
        ),
        "contradiction_signal": round(
            contradiction,
            4,
        ),
        "nli_stance": nli_stance,
        "nli_contradiction": round(
            nli_contradiction,
            4,
        ),
        "nli_entailment": round(
            nli_entailment,
            4,
        ),
        "nli_neutral": round(
            nli_neutral,
            4,
        ),
        "nli_confidence": round(
            nli_confidence,
            4,
        ),
        "reason": (
            "Evidence was not sufficiently relevant."
        ),
    }


# ============================================================
# MULTIPLE EVIDENCE ARTICLES
# ============================================================

def classify_evidence_articles(
    claim: str,
    articles: List[Dict],
) -> List[Dict]:
    """
    Classify every retrieved evidence article.

    Expected article fields:

        title
        description OR text
        similarity
    """

    results = []

    for article in articles:

        title = article.get(
            "title",
            "",
        )

        description = article.get(
            "description",
            "",
        )

        text = article.get(
            "text",
            "",
        )

        evidence_text = (
            f"{title}. "
            f"{description}. "
            f"{text}"
        )

        similarity = article.get(
            "similarity",
            0.0,
        )

        stance_result = classify_stance(
            claim=claim,
            evidence=evidence_text,
            similarity=similarity,
        )

        result = dict(article)
        result.update(stance_result)

        results.append(result)

    return results


# ============================================================
# STANCE SUMMARY
# ============================================================

def summarize_stances(
    classified_articles: List[Dict],
) -> Dict:
    """
    Summarize stance counts across all evidence articles.
    """

    counts = {
        "SUPPORTING": 0,
        "CONTRADICTING": 0,
        "RELATED": 0,
        "INSUFFICIENT": 0,
    }

    for article in classified_articles:

        stance = article.get(
            "stance",
            "INSUFFICIENT",
        )

        if stance in counts:
            counts[stance] += 1

    total = len(classified_articles)

    # Both supporting and contradicting evidence
    # means the evidence set is conflicting.
    if (
        counts["SUPPORTING"] > 0
        and counts["CONTRADICTING"] > 0
    ):
        overall = "CONFLICTING"

    elif counts["SUPPORTING"] > 0:
        overall = "SUPPORTING"

    elif counts["CONTRADICTING"] > 0:
        overall = "CONTRADICTING"

    elif counts["RELATED"] > 0:
        overall = "RELATED"

    else:
        overall = "INSUFFICIENT"

    return {
        "overall_stance": overall,
        "counts": counts,
        "total_articles": total,
    }


# ============================================================
# BASIC TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("EVIDENCE STANCE CLASSIFICATION TEST")
    print("=" * 70)

    tests = [
        {
            "name": "SUPPORTING",
            "claim": (
                "Gujarat Industrial Policy 2026 "
                "was unveiled to boost investment "
                "and economic growth."
            ),
            "evidence": (
                "Gujarat Industrial Policy 2026 "
                "was unveiled to boost investment "
                "and economic growth."
            ),
            "similarity": 0.95,
        },
        {
            "name": "CONTRADICTING",
            "claim": (
                "Gujarat Industrial Policy 2026 "
                "was unveiled to boost investment "
                "and economic growth."
            ),
            "evidence": (
                "Gujarat Industrial Policy 2026 "
                "was not unveiled and officials "
                "denied the announcement."
            ),
            "similarity": 0.82,
        },
        {
            "name": "RELATED",
            "claim": (
                "Gujarat Industrial Policy 2026 "
                "increased India's GDP by 10 percent."
            ),
            "evidence": (
                "Gujarat unveiled its Industrial "
                "Policy 2026 with a focus on "
                "investment and innovation."
            ),
            "similarity": 0.55,
        },
        {
            "name": "INSUFFICIENT",
            "claim": (
                "ZXQ-8472 government program was "
                "secretly launched yesterday."
            ),
            "evidence": (
                "Scientists discovered a new "
                "species of butterfly in South America."
            ),
            "similarity": 0.08,
        },
    ]

    for index, test in enumerate(
        tests,
        start=1,
    ):

        result = classify_stance(
            claim=test["claim"],
            evidence=test["evidence"],
            similarity=test["similarity"],
        )

        print()
        print("-" * 70)
        print(f"TEST {index}: {test['name']}")
        print("-" * 70)

        print("Expected:")
        print(test["name"])

        print()
        print("Actual:")
        print(result["stance"])

        print()
        print("Similarity:")
        print(result["similarity"])

        print("Keyword overlap:")
        print(result["keyword_overlap"])

        print("NLI stance:")
        print(result["nli_stance"])

        print("NLI contradiction:")
        print(result["nli_contradiction"])

        print("NLI entailment:")
        print(result["nli_entailment"])

        print("NLI neutral:")
        print(result["nli_neutral"])

        print("NLI confidence:")
        print(result["nli_confidence"])

        print()
        print("Reason:")
        print(result["reason"])

        expected = test["name"]
        actual = result["stance"]

        print()
        print(
            "PASS"
            if actual == expected
            else "FAIL"
        )

    print()
    print("=" * 70)
    print("STANCE CLASSIFICATION TEST COMPLETE")
    print("=" * 70)