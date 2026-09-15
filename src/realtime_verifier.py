"""
Enhanced Real-Time Evidence Verification
Day 17 - Evidence Verification Upgrade

This module evaluates retrieved evidence using multiple signals:

1. Semantic similarity
2. Keyword coverage
3. Source reliability
4. Freshness
5. Evidence diversity
6. Evidence coverage

IMPORTANT:
Evidence strength is NOT a probability that the claim is true.
It represents the quality and relevance of the retrieved evidence.
"""

from datetime import datetime, timezone
from urllib.parse import urlparse
import re


# ============================================================
# SOURCE RELIABILITY
# ============================================================

HIGH_QUALITY_SOURCES = {
    "reuters",
    "bbc",
    "associated press",
    "ap news",
    "the hindu",
    "the indian express",
    "pib",
    "press information bureau",
    "government of india",
    "cmo gujarat",
    "economic times",
    "times of india",
    "ndtv",
}

MEDIUM_QUALITY_SOURCES = {
    "hindustan times",
    "india today",
    "business standard",
    "financial express",
    "mint",
    "news18",
    "deccan herald",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "of", "to", "in", "on", "for", "with", "by", "from",
    "at", "as", "is", "was", "are", "were", "be", "been",
    "this", "that", "these", "those", "it", "its",
    "has", "have", "had", "will", "would", "can", "could",
    "may", "might", "should", "do", "does", "did",
    "about", "into", "after", "before", "over", "under",
    "during", "than", "also", "very", "more", "most"
}


def normalize_text(text):
    """
    Convert text into a normalized lowercase representation.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_keywords(text):
    """
    Extract informative keywords while removing common stopwords.
    """

    normalized = normalize_text(text)

    words = normalized.split()

    keywords = {
        word
        for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }

    return keywords


# ============================================================
# KEYWORD COVERAGE
# ============================================================

def keyword_overlap(claim, article):
    """
    Calculate the proportion of claim keywords appearing
    in the evidence article.

    Returns:
        float in [0, 1]
    """

    claim_keywords = extract_keywords(claim)
    article_keywords = extract_keywords(article)

    if not claim_keywords:
        return 0.0

    overlap = claim_keywords.intersection(article_keywords)

    return len(overlap) / len(claim_keywords)


# ============================================================
# SOURCE QUALITY
# ============================================================

def source_quality(source):
    """
    Estimate source reliability using a transparent heuristic.

    Returns:
        1.0  -> high-quality source
        0.7  -> medium-quality source
        0.4  -> unknown source
    """

    if not source:
        return 0.4

    source_normalized = normalize_text(source)

    for high_source in HIGH_QUALITY_SOURCES:
        if normalize_text(high_source) in source_normalized:
            return 1.0

    for medium_source in MEDIUM_QUALITY_SOURCES:
        if normalize_text(medium_source) in source_normalized:
            return 0.7

    return 0.4


# ============================================================
# FRESHNESS
# ============================================================

def freshness_score(published):
    """
    Estimate temporal relevance of evidence.

    Returns a score in [0, 1].
    """

    if not published:
        return 0.3

    try:
        published_dt = datetime.strptime(
            published,
            "%a, %d %b %Y %H:%M:%S %Z"
        )

        published_dt = published_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        age_days = max(
            0,
            (now - published_dt).total_seconds() / 86400
        )

        if age_days <= 1:
            return 1.0

        if age_days <= 7:
            return 0.9

        if age_days <= 30:
            return 0.8

        if age_days <= 90:
            return 0.6

        if age_days <= 365:
            return 0.4

        return 0.2

    except Exception:
        return 0.3


# ============================================================
# SOURCE DIVERSITY
# ============================================================

def extract_domain(url):
    """
    Extract normalized domain from an article URL.
    """

    if not url:
        return ""

    try:
        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


def calculate_source_diversity(articles):
    """
    Measure diversity of evidence sources.

    More independent domains produce a stronger diversity signal.

    Returns:
        float in [0, 1]
    """

    if not articles:
        return 0.0

    domains = set()

    for article in articles:
        domain = extract_domain(article.get("url", ""))

        if domain:
            domains.add(domain)

        else:
            source = normalize_text(
                article.get("source", "")
            )

            if source:
                domains.add(source)

    if not domains:
        return 0.0

    # Saturates at five distinct sources.
    return min(len(domains) / 5.0, 1.0)


# ============================================================
# EVIDENCE COVERAGE
# ============================================================

def calculate_evidence_coverage(claim, articles):
    """
    Measure how much of the claim's keyword content is
    covered across the retrieved evidence.

    Unlike keyword_overlap(), this considers the entire
    evidence set instead of only one article.

    Returns:
        float in [0, 1]
    """

    claim_keywords = extract_keywords(claim)

    if not claim_keywords or not articles:
        return 0.0

    covered_keywords = set()

    for article in articles:

        article_text = " ".join([
            article.get("title", ""),
            article.get("description", ""),
            article.get("text", "")
        ])

        article_keywords = extract_keywords(article_text)

        covered_keywords.update(
            claim_keywords.intersection(article_keywords)
        )

    return len(
        covered_keywords.intersection(claim_keywords)
    ) / len(claim_keywords)


# ============================================================
# INDIVIDUAL EVIDENCE SCORE
# ============================================================

def calculate_evidence_score(
    similarity,
    overlap,
    source,
    published
):
    """
    Calculate the quality score of one evidence article.

    Components:

        40% semantic similarity
        25% keyword overlap
        20% source quality
        15% freshness

    Returns:
        float in [0, 1]

    IMPORTANT:
    This is an evidence-quality score, not truth probability.
    """

    similarity = max(
        0.0,
        min(float(similarity), 1.0)
    )

    overlap = max(
        0.0,
        min(float(overlap), 1.0)
    )

    source_score = source_quality(source)

    freshness = freshness_score(published)

    score = (
        0.40 * similarity
        + 0.25 * overlap
        + 0.20 * source_score
        + 0.15 * freshness
    )

    return round(score, 4)


# ============================================================
# ARTICLE VERIFICATION
# ============================================================

def verify_articles(claim, articles):
    """
    Verify and rank retrieved evidence articles.
    """

    if not articles:
        return []

    verified = []

    for article in articles:

        article_text = " ".join([
            article.get("title", ""),
            article.get("description", ""),
            article.get("text", "")
        ])

        similarity = float(
            article.get("similarity", 0.0)
        )

        overlap = keyword_overlap(
            claim,
            article_text
        )

        source_score = source_quality(
            article.get("source", "")
        )

        freshness = freshness_score(
            article.get("published", "")
        )

        evidence_score = calculate_evidence_score(
            similarity=similarity,
            overlap=overlap,
            source=article.get("source", ""),
            published=article.get("published", "")
        )

        verified_article = dict(article)

        verified_article["keyword_overlap"] = round(
            overlap,
            4
        )

        verified_article["source_quality"] = round(
            source_score,
            4
        )

        verified_article["freshness_score"] = round(
            freshness,
            4
        )

        verified_article["evidence_score"] = evidence_score

        verified.append(
            verified_article
        )

    verified.sort(
        key=lambda item: item["evidence_score"],
        reverse=True
    )

    return verified


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

def summarize_evidence(claim, articles):
    """
    Produce an aggregate evidence assessment.

    The aggregate score combines:

        45% strongest evidence
        20% evidence coverage
        15% source diversity
        20% average evidence quality

    This reduces dependence on a single article.
    """

    if not articles:
        return {
            "evidence_signal": "INSUFFICIENT",
            "evidence_strength": 0.0,
            "articles_evaluated": 0,
            "source_diversity": 0.0,
            "evidence_coverage": 0.0,
            "average_evidence_score": 0.0
        }

    verified_articles = verify_articles(
        claim,
        articles
    )

    scores = [
        article["evidence_score"]
        for article in verified_articles
    ]

    strongest_score = max(scores)

    average_score = sum(scores) / len(scores)

    coverage = calculate_evidence_coverage(
        claim,
        verified_articles
    )

    diversity = calculate_source_diversity(
        verified_articles
    )

    evidence_strength = (
        0.45 * strongest_score
        + 0.20 * coverage
        + 0.15 * diversity
        + 0.20 * average_score
    )

    evidence_strength = round(
        max(
            0.0,
            min(evidence_strength, 1.0)
        ),
        4
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
        "evidence_signal": signal,
        "evidence_strength": evidence_strength,
        "articles_evaluated": len(verified_articles),
        "source_diversity": round(diversity, 4),
        "evidence_coverage": round(coverage, 4),
        "average_evidence_score": round(
            average_score,
            4
        ),
        "strongest_evidence_score": round(
            strongest_score,
            4
        )
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    claim = (
        "The government announced a new economic policy "
        "to increase employment."
    )

    sample_articles = [
        {
            "title": (
                "Government announces new economic policy "
                "to increase employment"
            ),
            "description": (
                "Officials said the policy aims to create "
                "new employment opportunities."
            ),
            "text": (
                "Government officials announced a new "
                "economic policy focused on employment."
            ),
            "source": "Reuters",
            "published": "Mon, 14 Sep 2026 08:00:00 GMT",
            "url": "https://www.reuters.com/example",
            "similarity": 0.91
        },
        {
            "title": (
                "Economic policy expected to boost jobs"
            ),
            "description": (
                "The new policy could support employment "
                "and economic activity."
            ),
            "text": (
                "Officials discussed employment and "
                "economic growth."
            ),
            "source": "BBC",
            "published": "Sun, 13 Sep 2026 10:00:00 GMT",
            "url": "https://www.bbc.com/example",
            "similarity": 0.82
        }
    ]

    verified = verify_articles(
        claim,
        sample_articles
    )

    summary = summarize_evidence(
        claim,
        sample_articles
    )

    print("=" * 70)
    print("DAY 17 - EVIDENCE VERIFICATION TEST")
    print("=" * 70)

    print("\nVERIFIED ARTICLES")

    for index, article in enumerate(
        verified,
        start=1
    ):
        print(f"\nArticle {index}")
        print("Source:", article["source"])
        print(
            "Similarity:",
            article["similarity"]
        )
        print(
            "Keyword overlap:",
            article["keyword_overlap"]
        )
        print(
            "Source quality:",
            article["source_quality"]
        )
        print(
            "Freshness:",
            article["freshness_score"]
        )
        print(
            "Evidence score:",
            article["evidence_score"]
        )

    print("\nAGGREGATE EVIDENCE")

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print("\nDAY 17 VERIFIER TEST COMPLETE")