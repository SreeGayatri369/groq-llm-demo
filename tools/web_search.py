"""
Web Search Tool — uses Tavily Search API for high-quality web results.
"""
import requests
from config import TAVILY_API_KEY


def web_search(query: str) -> str:
    """
    Search the web using Tavily API.

    Args:
        query: Search query string.

    Returns:
        Formatted search result string.
    """
    if not TAVILY_API_KEY:
        return (
            f"Tavily API key not configured. "
            f"Add TAVILY_API_KEY to your .env file (free at tavily.com). "
            f"Query: {query}"
        )

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": 3,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return f"No web results found for: {query}"

        parts = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            parts.append(f"**{title}**\n{content}\n{url}")

        return "\n\n".join(parts)

    except requests.HTTPError as e:
        return f"Tavily API error ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Web search error: {e}"
