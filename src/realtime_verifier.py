from datetime import datetime, timezone
import re
from urllib.parse import urlparse


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

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "of", "to", "in", "on", "for", "with", "by", "from",
    "at", "as", "is", "was", "are", "were", "be", "been",
    "this", "that", "these", "those", "it", "its",
    "has", "have", "had", "will", "would", "can", "could",
    "may", "might", "should", "do", "does", "did",
    "about", "into", "after", "before", "over", "under",
    "during", "than", "also", "very", "more", "most",
}


def normalize_text(text):
    """Lowercase text and remove URLs, punctuation, and extra spaces."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords(text):
    """Return a set of informative words from the supplied text."""
    words = normalize_text(text).split()
    return {
        word for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }


def keyword_overlap(claim, article):
    """Return the fraction of claim keywords found in an article."""
    claim_keywords = extract_keywords(claim)
    if not claim_keywords:
        return 0.0

    article_keywords = extract_keywords(article)
    return len(claim_keywords & article_keywords) / len(claim_keywords)


def source_quality(source):
    """
    Estimate source quality using a simple, transparent heuristic.

    Scores: 1.0 for a listed high-quality source, 0.7 for a medium
    source, and 0.4 when the source is unknown.
    """
    normalized_source = normalize_text(source)
    if not normalized_source:
        return 0.4

    for name in HIGH_QUALITY_SOURCES:
        if normalize_text(name) in normalized_source:
            return 1.0

    for name in MEDIUM_QUALITY_SOURCES:
        if normalize_text(name) in normalized_source:
            return 0.7

    return 0.4


def freshness_score(published):
    """Estimate freshness from a supported RFC-style publication date."""
    if not published:
        return 0.3

    try:
        published_at = datetime.strptime(
            published, "%a, %d %b %Y %H:%M:%S %Z"
        ).replace(tzinfo=timezone.utc)

        age_days = max(
            0.0,
            (datetime.now(timezone.utc) - published_at).total_seconds()
            / 86400,
        )

    except (TypeError, ValueError):
        return 0.3

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


def extract_domain(url):
    """Extract a normalized hostname from an article URL."""
    if not url:
        return ""

    try:
        domain = urlparse(url).netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except (TypeError, ValueError):
        return ""


def calculate_source_diversity(articles):
    """
    Measure the number of distinct source domains.

    The score reaches 1.0 at five distinct sources. This is a heuristic,
    not a guarantee that the sources are editorially independent.
    """
    if not articles:
        return 0.0

    sources = set()

    for article in articles:
        domain = extract_domain(article.get("url", ""))
        if domain:
            sources.add(domain)
            continue

        source_name = normalize_text(article.get("source", ""))
        if source_name:
            sources.add(source_name)

    return min(len(sources) / 5.0, 1.0)


def calculate_evidence_coverage(claim, articles):
    """Measure how many claim keywords appear across the evidence set."""
    claim_keywords = extract_keywords(claim)
    if not claim_keywords or not articles:
        return 0.0

    covered_keywords = set()

    for article in articles:
        article_text = " ".join(
            (
                article.get("title", "") or "",
                article.get("description", "") or "",
                article.get("text", "") or "",
            )
        )
        covered_keywords.update(
            claim_keywords & extract_keywords(article_text)
        )

    return len(covered_keywords & claim_keywords) / len(claim_keywords)


def calculate_evidence_score(similarity, overlap, source, published):
    """
    Calculate an article evidence-quality score.

    Weights: 40% semantic similarity, 25% keyword overlap,
    20% source heuristic, and 15% freshness.

    This score does not determine whether a claim is true.
    """
    similarity = max(0.0, min(float(similarity), 1.0))
    overlap = max(0.0, min(float(overlap), 1.0))

    score = (
        0.40 * similarity
        + 0.25 * overlap
        + 0.20 * source_quality(source)
        + 0.15 * freshness_score(published)
    )
    return round(score, 4)


def verify_articles(claim, articles):
    """Add evidence-quality signals to each article and rank the results."""
    verified_articles = []

    for article in articles or []:
        article_text = " ".join(
            (
                article.get("title", "") or "",
                article.get("description", "") or "",
                article.get("text", "") or "",
            )
        )

        similarity = article.get("similarity", 0.0)
        try:
            similarity = float(similarity)
        except (TypeError, ValueError):
            similarity = 0.0

        overlap = keyword_overlap(claim, article_text)
        source = article.get("source", "")
        published = article.get("published", "")

        verified_article = {
            **article,
            "keyword_overlap": round(overlap, 4),
            "source_quality": round(source_quality(source), 4),
            "freshness_score": round(freshness_score(published), 4),
            "evidence_score": calculate_evidence_score(
                similarity=similarity,
                overlap=overlap,
                source=source,
                published=published,
            ),
        }
        verified_articles.append(verified_article)

    verified_articles.sort(
        key=lambda item: item["evidence_score"],
        reverse=True,
    )
    return verified_articles


def summarize_evidence(claim, articles):
    """
    Return an aggregate evidence assessment.

    The summary combines the strongest article (45%), keyword coverage
    (20%), source diversity (15%), and average article quality (20%).
    """
    verified_articles = verify_articles(claim, articles)

    if not verified_articles:
        return {
            "evidence_signal": "INSUFFICIENT",
            "evidence_strength": 0.0,
            "articles_evaluated": 0,
            "source_diversity": 0.0,
            "evidence_coverage": 0.0,
            "average_evidence_score": 0.0,
            "strongest_evidence_score": 0.0,
        }

    scores = [
        article["evidence_score"] for article in verified_articles
    ]
    strongest_score = max(scores)
    average_score = sum(scores) / len(scores)
    coverage = calculate_evidence_coverage(claim, verified_articles)
    diversity = calculate_source_diversity(verified_articles)

    strength = (
        0.45 * strongest_score
        + 0.20 * coverage
        + 0.15 * diversity
        + 0.20 * average_score
    )
    strength = round(max(0.0, min(strength, 1.0)), 4)

    if strength >= 0.70:
        signal = "STRONG"
    elif strength >= 0.55:
        signal = "MODERATE"
    elif strength >= 0.40:
        signal = "WEAK"
    else:
        signal = "INSUFFICIENT"

    return {
        "evidence_signal": signal,
        "evidence_strength": strength,
        "articles_evaluated": len(verified_articles),
        "source_diversity": round(diversity, 4),
        "evidence_coverage": round(coverage, 4),
        "average_evidence_score": round(average_score, 4),
        "strongest_evidence_score": round(strongest_score, 4),
    }


if __name__ == "__main__":
    claim = (
        "The government announced a new economic policy "
        "to increase employment."
    )

    sample_articles = [
        {
            "title": "Government announces new economic policy to increase employment",
            "description": "Officials said the policy aims to create new employment opportunities.",
            "text": "Government officials announced a new economic policy focused on employment.",
            "source": "Reuters",
            "published": "Mon, 14 Sep 2026 08:00:00 GMT",
            "url": "https://www.reuters.com/example",
            "similarity": 0.91,
        },
        {
            "title": "Economic policy expected to boost jobs",
            "description": "The new policy could support employment and economic activity.",
            "text": "Officials discussed employment and economic growth.",
            "source": "BBC",
            "published": "Sun, 13 Sep 2026 10:00:00 GMT",
            "url": "https://www.bbc.com/example",
            "similarity": 0.82,
        },
    ]

    verified = verify_articles(claim, sample_articles)
    summary = summarize_evidence(claim, sample_articles)

    print("=" * 65)
    print("EVIDENCE VERIFICATION TEST")
    print("=" * 65)

    print("\nVERIFIED ARTICLES")
    for index, article in enumerate(verified, start=1):
        print(f"\nArticle {index}: {article.get('title', 'Untitled')}")
        print(f"Source: {article.get('source', 'Unknown')}")
        print(f"Similarity: {article.get('similarity', 0.0)}")
        print(f"Keyword overlap: {article['keyword_overlap']}")
        print(f"Source quality: {article['source_quality']}")
        print(f"Freshness: {article['freshness_score']}")
        print(f"Evidence score: {article['evidence_score']}")

    print("\nAGGREGATE EVIDENCE")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nEvidence verification test complete.")
