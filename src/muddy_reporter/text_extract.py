from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup


_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def read_doc_text(local_path: str) -> str:
    p = Path(local_path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    # Heuristic: filings are usually HTML; keep it simple.
    if "<html" in raw.lower() or "<body" in raw.lower() or "<div" in raw.lower():
        return html_to_text(raw)
    return raw.strip()


def chunk_text(text: str, max_chars: int = 9000) -> list[str]:
    """
    Chunk by paragraph boundaries to stay within typical LLM context windows.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if size + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

