```python
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer


# Make the project source directory importable.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from realtime_evidence_retriever import retrieve_realtime_evidence


MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "all-MiniLM-L6-v2"
)

DEFAULT_TOP_K = 5
MIN_SIMILARITY = 0.30


def load_embedding_model():
    """Load the local sentence embedding model."""
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Embedding model directory not found: {MODEL_PATH}"
        )

    print("Loading sentence embedding model...")
    model = SentenceTransformer(MODEL_PATH)
    print("Embedding model ready.")

    return model


embedding_model = load_embedding_model()


def cosine_similarity(vector_a, vector_b):
    """Return cosine similarity between two vectors."""
    vector_a = np.asarray(vector_a, dtype=np.float32)
    vector_b = np.asarray(vector_b, dtype=np.float32)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    denominator = norm_a * norm_b

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def score_articles(claim, articles):
    """Rank retrieved articles by semantic similarity to a claim."""
    if not articles:
        return []

    claim_vector = embedding_model.encode(
        claim,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    article_texts = [
        article.get("text", "") or ""
        for article in articles
    ]

    article_vectors = embedding_model.encode(
        article_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    ranked_articles = []

    for article, article_vector in zip(
        articles, article_vectors
    ):
        similarity = cosine_similarity(
            claim_vector, article_vector
        )

        scored_article = {
            **article,
            "similarity": round(similarity, 4),
            "retrieval_type": "REAL_TIME_SEMANTIC",
        }

        ranked_articles.append(scored_article)

    ranked_articles.sort(
        key=lambda article: article["similarity"],
        reverse=True,
    )

    return ranked_articles


def retrieve_realtime_semantic_evidence(
    claim,
    top_k=DEFAULT_TOP_K,
    min_similarity=MIN_SIMILARITY,
):
    """Fetch current evidence and return the strongest matches."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim must be a non-empty string.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1.")

    articles = retrieve_realtime_evidence(
        claim,
        max_results=max(top_k * 2, 10),
    )

    if not articles:
        return []

    ranked_articles = score_articles(claim, articles)

    relevant_articles = [
        article
        for article in ranked_articles
        if article["similarity"] >= min_similarity
    ]

    return relevant_articles[:top_k]


def print_results(claim, results):
    """Display retrieved evidence in a readable format."""
    print("\n" + "=" * 65)
    print("REAL-TIME EVIDENCE RETRIEVAL")
    print("=" * 65)

    print(f"\nClaim: {claim}")
    print(f"Matching articles: {len(results)}")

    if not results:
        print("\nNo articles met the similarity threshold.")
        print("=" * 65)
        return

    for index, article in enumerate(results, start=1):
        print(f"\n[{index}] {article.get('title', 'Untitled')}")
        print(f"Source: {article.get('source', 'Unknown')}")
        print(f"Published: {article.get('published', 'Unknown')}")
        print(f"Similarity: {article['similarity']:.4f}")
        print(f"URL: {article.get('url', 'Unavailable')}")

        description = article.get("description", "")
        if description:
            print(f"Description: {description[:300]}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    claim = "India government announces new economic policy"

    print("\nSearching for relevant real-time evidence...")

    results = retrieve_realtime_semantic_evidence(
        claim,
        top_k=5,
    )

    print_results(claim, results)
```
