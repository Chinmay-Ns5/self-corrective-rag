import json
import re
import gzip
from html.parser import HTMLParser
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class TavilySearch:
    def __init__(self, api_key):
        self.api_key = api_key

    def _search_rows(self, query, max_results):
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is required for web fallback")
        payload = json.dumps({"query": query, "max_results": max_results,
                              "search_depth": "basic", "include_answer": False}).encode("utf-8")
        request = Request("https://api.tavily.com/search", data=payload,
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=20) as response:
            return json.load(response).get("results", [])

    def search(self, query, max_results=8):
        results = self._search_rows(query, max_results)
        evidence = [{"id": f"W{i}", "title": row.get("title", ""),
                 "url": row.get("url", ""), "text": row.get("content", ""),
                 "published_date": row.get("published_date", ""),
                 "score": row.get("score", 0)}
                for i, row in enumerate(results, start=1)]
        if re.search(r"\b(latest|newest|current)\b", query, re.I):
            latest = self._github_latest(query, evidence)
            if latest:
                evidence.append({"id": f"W{len(evidence) + 1}", **latest})
            elif re.search(r"\b(release|version)\b", query, re.I):
                official = self._official_release_page(query, evidence)
                if official:
                    evidence.append({"id": f"W{len(evidence) + 1}", **official})
        return evidence

    @staticmethod
    def _official_release_page(query, results):
        """Read a matching product's release page without another Tavily call."""
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        for row in results:
            url = row.get("url", "")
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()
            labels = host.split(".")
            if len(labels) < 2 or not re.search(r"download|release", parts.path, re.I):
                continue
            product = labels[-2]
            if product not in query_terms or len(product) < 4:
                continue
            try:
                request = Request(url, headers={"User-Agent": "self-corrective-rag"})
                with urlopen(request, timeout=8) as response:
                    raw = response.read(1_000_000)
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                parser = _VisibleText()
                parser.feed(raw.decode("utf-8", errors="replace"))
                visible = re.sub(r"\s+", " ", " ".join(parser.parts))
                match = re.search(r"\blatest\b.{0,180}?\b\d+\.\d+\.\d+\b", visible, re.I)
                if match and not re.search(r"\b(?:alpha|beta|preview|rc\d+)\b", match.group(), re.I):
                    return {"title": f"Official {product} release page",
                            "url": url, "text": visible[max(0, match.start() - 50):match.end() + 150],
                            "published_date": "", "score": 1.0}
            except (OSError, ValueError):
                continue
        return None

    @staticmethod
    def _github_latest(query, results):
        """Check a matching project's official GitHub latest-release endpoint."""
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        for row in results:
            parts = urlsplit(row.get("url", ""))
            match = re.fullmatch(r"/([^/]+)/([^/]+)/releases/?", parts.path)
            if parts.netloc.lower() != "github.com" or not match:
                continue
            owner, repo = match.groups()
            if not set(re.findall(r"[a-z0-9]+", repo.lower())) <= query_terms:
                continue
            api_url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/releases/latest"
            request = Request(api_url, headers={"Accept": "application/vnd.github+json",
                                                "User-Agent": "self-corrective-rag"})
            try:
                with urlopen(request, timeout=8) as response:
                    release = json.load(response)
                version = release["tag_name"]
                url = release["html_url"]
                published = release["published_at"]
                if not version or not url.startswith("https://github.com/"):
                    continue
                return {"title": f"Official {repo} latest stable release: {version}",
                        "url": url, "published_date": published,
                        "text": f"GitHub lists {version} as the latest stable release of {repo}, published {published}.",
                        "score": 1.0}
            except (OSError, ValueError, KeyError):
                continue
        return None


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
