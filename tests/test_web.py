import io
import json
import unittest
from unittest.mock import patch

from src.crag.web import TavilySearch


class WebTests(unittest.TestCase):
    def test_latest_release_uses_one_tavily_search(self):
        rows = [{"title": "Official downloads", "url": "https://zephyr.org/downloads",
                 "content": "Release list", "score": 0.4}]
        official = {"title": "Official release", "url": "https://zephyr.org/downloads",
                    "text": "Latest Zephyr 2.4.7", "score": 1.0}
        search = TavilySearch("test-key")
        with patch.object(search, "_search_rows", return_value=rows) as tavily, \
                patch.object(search, "_official_release_page", return_value=official):
            evidence = search.search("latest stable Zephyr release")
        tavily.assert_called_once()
        self.assertEqual(evidence[-1]["text"], "Latest Zephyr 2.4.7")

    def test_latest_release_uses_matching_official_repository(self):
        rows = [
            {"url": "https://github.com/other/project/releases"},
            {"url": "https://github.com/team/zephyr/releases"},
        ]
        release = {"tag_name": "v2.4.7", "published_at": "2026-10-02T00:00:00Z",
                   "html_url": "https://github.com/team/zephyr/releases/tag/v2.4.7"}
        with patch("src.crag.web.urlopen", return_value=io.BytesIO(json.dumps(release).encode())) as opened:
            result = TavilySearch._github_latest("latest stable Zephyr release", rows)
        self.assertEqual(result["published_date"], release["published_at"])
        self.assertIn("v2.4.7", result["text"])
        self.assertEqual(opened.call_count, 1)

    def test_unrelated_repository_is_ignored(self):
        with patch("src.crag.web.urlopen") as opened:
            result = TavilySearch._github_latest(
                "latest stable Zephyr release",
                [{"url": "https://github.com/other/project/releases"}],
            )
        self.assertIsNone(result)
        opened.assert_not_called()

    def test_official_release_page_extracts_latest_full_version_without_search(self):
        html = b"<html><script>version='0.1.0'</script><body>Download the latest version of Zephyr: 2.4.7</body></html>"
        with patch("src.crag.web.urlopen", return_value=io.BytesIO(html)) as opened:
            result = TavilySearch._official_release_page(
                "latest stable Zephyr release",
                [{"url": "https://zephyr.org/downloads"}],
            )
        self.assertIn("2.4.7", result["text"])
        self.assertEqual(opened.call_count, 1)


if __name__ == "__main__":
    unittest.main()
