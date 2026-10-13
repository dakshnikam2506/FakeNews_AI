```python
import os
import sys
import re
import html
import urllib.parse
import xml.etree.ElementTree as ET

from datetime import datetime, timezone

import requests


# ------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ------------------------------------------------------------
# SETTINGS
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
# CLEAN ARTICLE TEXT
# ------------------------------------------------------------

def clean_text(text):
    """Remove HTML tags, decode entities and trim whitespace."""

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
# PREPARE THE SEARCH URL
# ------------------------------------------------------------

def build_search_url(query):
    """Create the Google News RSS URL for a search query."""

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
# GET THE RSS RESPONSE
# ------------------------------------------------------------

def fetch_rss(query):
    """Request news results and return the RSS response."""

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
# EXTRACT ARTICLES FROM RSS
# ------------------------------------------------------------

def parse_rss(xml_text, max_results=DEFAULT_MAX_RESULTS):
    """Read the RSS feed and collect the available article details."""

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

        # Keep the full headline. Google News may include
        # the publisher's name as part of the title.

        results.append({
            "title": title,
            "description": description,
            "source": source,
            "published": pub_date,
            "url": link
        })

    return results


# ------------------------------------------------------------
# SEARCH FOR RECENT NEWS
# ------------------------------------------------------------

def search_realtime_news(
    query,
    max_results=DEFAULT_MAX_RESULTS
):
    """
    Search Google News for articles related to the query.

    Returns article details in a consistent dictionary format.
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
# STANDARDIZE ARTICLE DATA
# ------------------------------------------------------------

def normalize_article(article):
    """
    Prepare an article for use by the retrieval pipeline.

    The combined text field is used later when comparing
    the article with the claim being checked.
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
# RETRIEVE EVIDENCE FOR A CLAIM
# ------------------------------------------------------------

def retrieve_realtime_evidence(
    claim,
    max_results=DEFAULT_MAX_RESULTS
):
    """
    Find news articles related to a claim and standardize
    their fields before passing them to later pipeline stages.
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
# DISPLAY SEARCH RESULTS
# ------------------------------------------------------------

def print_results(claim, results):

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
# RUN A QUICK RETRIEVAL TEST
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
```
