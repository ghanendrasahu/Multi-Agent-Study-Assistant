"""
tools/search.py
───────────────
Search tools available to the Researcher agent.

Concepts from document:
  - Tool categories: Search tools (section 4a of tool categories)
  - Tool error handling: graceful fallback if API unavailable
  - Tool docstrings: the docstring IS the tool's interface — the LLM reads it
  - Retry logic: network failures are retried before returning a fallback

All tools return plain strings — easy for the LLM to consume.
"""

from __future__ import annotations

import time
from typing import Optional


def wikipedia_search(query: str, sentences: int = 10) -> str:
    """
    Search Wikipedia for a factual overview of a topic.

    Args:
        query: The search query (topic name or question)
        sentences: Number of summary sentences to return (default 10)

    Returns:
        A text summary from Wikipedia, or an error message if not found.

    Tool design principle: always return a string, never raise.
    """
    try:
        import wikipediaapi

        wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="StudyAssistant/1.0 (subrat-study-agent)"
        )

        page = wiki.page(query)
        if page.exists():
            # Return first N sentences of the summary
            summary = page.summary
            sentences_list = summary.split(". ")
            truncated = ". ".join(sentences_list[:sentences])
            return f"[Wikipedia: {page.title}]\n{truncated}..."

        # Try a broader search if exact page not found
        return f"[Wikipedia] No exact page found for '{query}'. " \
               f"Consider searching for a more specific term."

    except ImportError:
        return "[Wikipedia] wikipedia-api not installed. Run: pip install wikipedia-api"
    except Exception as e:
        return f"[Wikipedia] Search failed: {str(e)}"


def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo for recent information.

    Args:
        query: Search query
        max_results: Number of results to return (default 3)

    Returns:
        Formatted string of search results (title + snippet + URL).

    Note: DuckDuckGo requires no API key — ideal for demos.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"Snippet: {r.get('body', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}"
                )
                time.sleep(0.1)  # Polite rate limiting

        if results:
            return "[DuckDuckGo Results]\n\n" + "\n\n---\n\n".join(results)
        else:
            return "[DuckDuckGo] No results found."

    except ImportError:
        return "[DuckDuckGo] duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"[DuckDuckGo] Search failed: {str(e)}"


def combined_search(query: str) -> str:
    """
    Runs both Wikipedia and DuckDuckGo searches and combines results.
    This is the default tool used by the Researcher agent.

    Args:
        query: The research query

    Returns:
        Combined search results from both sources.
    """
    wiki = wikipedia_search(query)
    web = duckduckgo_search(query)
    return f"{wiki}\n\n{web}"
