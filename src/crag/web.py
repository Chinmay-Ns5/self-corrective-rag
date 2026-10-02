import json
from urllib.request import Request, urlopen


class TavilySearch:
    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, query, max_results=5):
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is required for web fallback")
        payload = json.dumps({"query": query, "max_results": max_results,
                              "search_depth": "basic", "include_answer": False}).encode("utf-8")
        request = Request("https://api.tavily.com/search", data=payload,
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=20) as response:
            results = json.load(response).get("results", [])
        return [{"id": f"W{i}", "title": row.get("title", ""),
                 "url": row.get("url", ""), "text": row.get("content", ""),
                 "score": row.get("score", 0)}
                for i, row in enumerate(results, start=1)]


def refine_web(results, limit=4, max_chars=1200):
    """Keep relevant, distinct, attributable snippets within a bounded context."""
    refined, seen = [], set()
    for row in sorted(results, key=lambda item: item.get("score", 0), reverse=True):
        url = (row.get("url") or "").strip()
        text = (row.get("text") or "").strip()
        if not url.startswith(("https://", "http://")) or not text or url in seen:
            continue
        seen.add(url)
        refined.append({**row, "text": text[:max_chars]})
        if len(refined) == limit:
            break
    return refined
