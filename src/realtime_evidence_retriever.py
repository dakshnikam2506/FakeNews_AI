import os
import sys
import re
import html
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests


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
# CONFIGURATION
# ------------------------------------------------------------

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search"
)

DEFAULT_MAX_RESULTS = 10

REQUEST_TIMEOUT = 15

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


# ------------------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    """
    Clean HTML entities and unnecessary whitespace.
    """

    if text is None:
        return ""

    text = html.unescape(str(text))

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ------------------------------------------------------------
# GOOGLE NEWS RSS URL
# ------------------------------------------------------------

def build_search_url(query):
    """
    Build a Google News RSS search URL.
    """

    encoded_query = urllib.parse.quote_plus(
        query
    )

    return (
        f"{GOOGLE_NEWS_RSS_URL}"
        f"?q={encoded_query}"
        f"&hl=en-IN"
        f"&gl=IN"
        f"&ceid=IN:en"
    )


# ------------------------------------------------------------
# FETCH RSS
# ------------------------------------------------------------

def fetch_rss(query):
    """
    Fetch Google News RSS results for a query.
    """

    url = build_search_url(query)

    headers = {
        "User-Agent": USER_AGENT
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.text


# ------------------------------------------------------------
# PARSE RSS
# ------------------------------------------------------------

def parse_rss(xml_text, max_results=DEFAULT_MAX_RESULTS):
    """
    Parse RSS XML and extract news articles.
    """

    root = ET.fromstring(
        xml_text
    )

    results = []

    channel = root.find("channel")

    if channel is None:
        return results

    items = channel.findall("item")

    for item in items[:max_results]:

        title_element = item.find("title")
        link_element = item.find("link")
        description_element = item.find("description")
        pub_date_element = item.find("pubDate")
        source_element = item.find("source")

        title = (
            title_element.text
            if title_element is not None
            else ""
        )

        link = (
            link_element.text
            if link_element is not None
            else ""
        )

        description = (
            description_element.text
            if description_element is not None
            else ""
        )

        pub_date = (
            pub_date_element.text
            if pub_date_element is not None
            else ""
        )

        source = (
            source_element.text
            if source_element is not None
            else ""
        )

        title = clean_text(title)
        description = clean_text(description)
        source = clean_text(source)
        link = link.strip() if link else ""

        # ----------------------------------------------------
        # Google News titles often look like:
        #
        # "Headline - BBC"
        #
        # Keep the complete title because it can contain
        # useful source information.
        # ----------------------------------------------------

        results.append({
            "title": title,
            "description": description,
            "source": source,
            "published": pub_date,
            "url": link
        })

    return results


# ------------------------------------------------------------
# REAL-TIME NEWS SEARCH
# ------------------------------------------------------------

def search_realtime_news(
    query,
    max_results=DEFAULT_MAX_RESULTS
):
    """
    Search current news using Google News RSS.

    Returns a list of dictionaries containing:

        title
        description
        source
        published
        url
    """

    if not query or not query.strip():

        return []

    try:

        xml_text = fetch_rss(
            query.strip()
        )

        results = parse_rss(
            xml_text,
            max_results=max_results
        )

        return results

    except requests.RequestException as e:

        print(
            f"[REALTIME SEARCH ERROR] "
            f"Network error: {e}"
        )

        return []

    except ET.ParseError as e:

        print(
            f"[REALTIME SEARCH ERROR] "
            f"RSS parsing error: {e}"
        )

        return []

    except Exception as e:

        print(
            f"[REALTIME SEARCH ERROR] "
            f"Unexpected error: {e}"
        )

        return []


# ------------------------------------------------------------
# NORMALIZE ARTICLE
# ------------------------------------------------------------

def normalize_article(article):
    """
    Create a normalized representation that can later be
    passed into the semantic retrieval / decision pipeline.
    """

    title = article.get(
        "title",
        ""
    )

    description = article.get(
        "description",
        ""
    )

    source = article.get(
        "source",
        ""
    )

    published = article.get(
        "published",
        ""
    )

    url = article.get(
        "url",
        ""
    )

    # Combine title and description for semantic comparison.
    combined_text = (
        f"{title}. {description}"
    ).strip()

    return {
        "title": title,
        "description": description,
        "source": source,
        "published": published,
        "url": url,
        "text": combined_text,
        "retrieval_type": "REAL_TIME"
    }


# ------------------------------------------------------------
# REAL-TIME EVIDENCE RETRIEVAL
# ------------------------------------------------------------

def retrieve_realtime_evidence(
    claim,
    max_results=DEFAULT_MAX_RESULTS
):
    """
    Retrieve fresh news articles relevant to a claim.

    This function currently performs retrieval only.

    Semantic similarity scoring and factual verification
    will be added in later Day 12 steps.
    """

    articles = search_realtime_news(
        claim,
        max_results=max_results
    )

    normalized = []

    for article in articles:

        normalized_article = normalize_article(
            article
        )

        normalized.append(
            normalized_article
        )

    return normalized


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

def print_results(
    claim,
    results
):

    print("\n")
    print("=" * 75)
    print("REAL-TIME NEWS RETRIEVAL TEST")
    print("=" * 75)

    print("\nCLAIM:")
    print(claim)

    print("\nRESULT COUNT:")
    print(len(results))

    print("\n" + "-" * 75)

    if not results:

        print(
            "No real-time news results were retrieved."
        )

        print("\n" + "=" * 75)

        return

    for index, result in enumerate(
        results,
        start=1
    ):

        print(f"\nRESULT {index}")

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
        "\nRunning FakeNews_AI real-time evidence retrieval..."
    )

    results = retrieve_realtime_evidence(
        test_claim,
        max_results=10
    )

    print_results(
        test_claim,
        results
    )