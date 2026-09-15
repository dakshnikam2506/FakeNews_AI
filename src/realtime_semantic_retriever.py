import os
import sys
import numpy as np

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from sentence_transformers import SentenceTransformer

from realtime_evidence_retriever import (
    retrieve_realtime_evidence
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "all-MiniLM-L6-v2"
)

DEFAULT_TOP_K = 5

# Minimum semantic similarity for an article to be considered
# potentially relevant evidence.
MIN_SIMILARITY = 0.30


# ------------------------------------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------------------------------------

print(
    "[INFO] Loading semantic embedding model..."
)

embedding_model = SentenceTransformer(
    MODEL_PATH
)

print(
    "[INFO] Embedding model loaded."
)


# ------------------------------------------------------------
# COSINE SIMILARITY
# ------------------------------------------------------------

def cosine_similarity(
    vector_a,
    vector_b
):
    """
    Calculate cosine similarity between two vectors.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / denominator
    )


# ------------------------------------------------------------
# SCORE REAL-TIME ARTICLES
# ------------------------------------------------------------

def score_articles(
    claim,
    articles
):
    """
    Calculate semantic similarity between the claim and
    each real-time article.
    """

    if not articles:
        return []

    # --------------------------------------------------------
    # Claim embedding
    # --------------------------------------------------------

    claim_embedding = embedding_model.encode(
        claim,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scored_articles = []

    # --------------------------------------------------------
    # Article embeddings
    # --------------------------------------------------------

    article_texts = [
        article.get("text", "")
        for article in articles
    ]

    article_embeddings = embedding_model.encode(
        article_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    for article, article_embedding in zip(
        articles,
        article_embeddings
    ):

        similarity = cosine_similarity(
            claim_embedding,
            article_embedding
        )

        result = dict(article)

        result["similarity"] = round(
            similarity,
            4
        )

        result["retrieval_type"] = (
            "REAL_TIME_SEMANTIC"
        )

        scored_articles.append(
            result
        )

    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    scored_articles.sort(
        key=lambda x: x.get(
            "similarity",
            0.0
        ),
        reverse=True
    )

    return scored_articles


# ------------------------------------------------------------
# RETRIEVE SEMANTIC REAL-TIME EVIDENCE
# ------------------------------------------------------------

def retrieve_realtime_semantic_evidence(
    claim,
    top_k=DEFAULT_TOP_K,
    min_similarity=MIN_SIMILARITY
):
    """
    Retrieve current news articles and rank them using
    semantic similarity.
    """

    articles = retrieve_realtime_evidence(
        claim,
        max_results=max(
            top_k * 2,
            10
        )
    )

    if not articles:
        return []

    scored_articles = score_articles(
        claim,
        articles
    )

    # --------------------------------------------------------
    # Filter weak results
    # --------------------------------------------------------

    filtered = [
        article
        for article in scored_articles
        if article.get(
            "similarity",
            0.0
        ) >= min_similarity
    ]

    # --------------------------------------------------------
    # Return top K
    # --------------------------------------------------------

    return filtered[:top_k]


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

def print_results(
    claim,
    results
):

    print("\n")
    print("=" * 75)
    print("REAL-TIME SEMANTIC EVIDENCE TEST")
    print("=" * 75)

    print("\nCLAIM:")
    print(claim)

    print("\nRESULT COUNT:")
    print(len(results))

    print("\n" + "-" * 75)

    if not results:

        print(
            "No sufficiently relevant real-time evidence found."
        )

        print("\n" + "=" * 75)

        return

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {index}"
        )

        print(
            f"Title: "
            f"{result.get('title', '')}"
        )

        print(
            f"Source: "
            f"{result.get('source', '')}"
        )

        print(
            f"Published: "
            f"{result.get('published', '')}"
        )

        print(
            f"Similarity: "
            f"{result.get('similarity', 0.0):.4f}"
        )

        print(
            f"URL: "
            f"{result.get('url', '')}"
        )

        description = result.get(
            "description",
            ""
        )

        if description:

            print(
                f"Description: "
                f"{description[:300]}"
            )

    print("\n" + "=" * 75)


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    test_claim = (
        "India government announces new economic policy"
    )

    print(
        "\nRunning FakeNews_AI semantic real-time retrieval..."
    )

    results = retrieve_realtime_semantic_evidence(
        test_claim,
        top_k=5
    )

    print_results(
        test_claim,
        results
    )