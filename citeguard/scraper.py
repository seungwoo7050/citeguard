from __future__ import annotations

import re

import httpx
import markdownify

NOISE_TAGS_RE = re.compile(
    r"<(script|style|nav|footer|noscript|svg|iframe)\b[\s\S]*?</\1>",
    re.IGNORECASE,
)


def html_to_markdown(raw_html: str) -> str:
    """Converts HTML to Markdown, stripping noise tags and preserving headings and tables."""
    if (
        "<html" not in raw_html.lower()
        and "<body" not in raw_html.lower()
        and "<h" not in raw_html.lower()
        and "<table" not in raw_html.lower()
        and "<p" not in raw_html.lower()
    ):
        return raw_html.strip()
    cleaned = NOISE_TAGS_RE.sub("", raw_html)
    md = markdownify.markdownify(cleaned, heading_style="ATX")
    return re.sub(r"\n{3,}", "\n\n", md).strip()


class LiveDocumentScraper:
    """Fetches live policy/regulation pages over HTTP and normalizes HTML or Markdown."""

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._transport = transport

    async def fetch_markdown(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            transport=self._transport,
            headers={"User-Agent": "CiteGuard-PolicyIndexer/0.1"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            text = resp.text
            if "html" in content_type or "<html" in text.lower() or "<table" in text.lower():
                return html_to_markdown(text)
            return text.strip()
