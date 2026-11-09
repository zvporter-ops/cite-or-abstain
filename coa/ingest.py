"""Text extraction for "anything you might give it".

Standard library only, except PDF and OCR, which use a binary when one is present and
skip cleanly when it is not. An extractor that silently returns an empty string is worse
than one that returns None: the index would record a document it cannot search.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".text", ".rst", ".org", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".html", ".htm", ".xml", ".log", ".ini", ".cfg",
    ".py", ".js", ".ts", ".sh", ".sql", ".go", ".rs", ".java", ".rb",
}
OFFICE_SUFFIXES = {".docx", ".xlsx", ".pptx"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

SUPPORTED = TEXT_SUFFIXES | OFFICE_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_XML_TEXT = re.compile(r"<(?:w:t|a:t|t)[^>]*>(.*?)</(?:w:t|a:t|t)>", re.DOTALL)


def supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


def extract_text(path: Path, max_bytes: int = 2_000_000) -> str | None:
    """Return searchable text for ``path``, or None when it cannot be read."""
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            return _read_text(path, max_bytes)
        if suffix in OFFICE_SUFFIXES:
            return _office_text(path)
        if suffix in PDF_SUFFIXES:
            return _pdf_text(path)
        if suffix in IMAGE_SUFFIXES:
            return _ocr_text(path)
    except (OSError, zipfile.BadZipFile, subprocess.SubprocessError):
        return None
    return None


def _read_text(path: Path, max_bytes: int) -> str | None:
    raw = path.read_bytes()[:max_bytes]
    if b"\x00" in raw[:4096]:
        return None  # binary masquerading as text
    text = raw.decode("utf-8", "replace")
    if path.suffix.lower() in {".html", ".htm"}:
        text = _strip_html(text)
    return text


def _strip_html(text: str) -> str:
    text = _SCRIPT.sub(" ", text)
    text = _TAG.sub(" ", text)
    return re.sub(r"[ \t]+", " ", html.unescape(text))


def _office_text(path: Path) -> str | None:
    """Word/PowerPoint/Excel: read the XML parts and keep their text runs."""
    with zipfile.ZipFile(path) as archive:
        parts = [n for n in archive.namelist()
                 if n.endswith(".xml") and ("document" in n or "slide" in n
                                            or "sheet" in n or "sharedStrings" in n)]
        chunks = []
        for name in parts:
            xml = archive.read(name).decode("utf-8", "replace")
            chunks.extend(m.group(1) for m in _XML_TEXT.finditer(xml))
    text = "\n".join(chunk for chunk in chunks if chunk.strip())
    return text or None


def _child_env() -> dict[str, str]:
    """A minimal environment for the extractor children: no secrets, just enough to run."""
    env = {
        "PATH": os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }
    home = os.environ.get("HOME")
    if home:
        env["HOME"] = home  # tesseract wants a writable home for its cache
    return env


def _pdf_text(path: Path) -> str | None:
    binary = shutil.which("pdftotext")
    if not binary:
        return None
    result = subprocess.run([binary, "-q", "-layout", str(path), "-"],
                            capture_output=True, timeout=60, env=_child_env())
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", "replace") or None


def _ocr_text(path: Path) -> str | None:
    binary = shutil.which("tesseract")
    if not binary:
        return None
    result = subprocess.run([binary, str(path), "stdout"],
                            capture_output=True, timeout=120, env=_child_env())
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", "replace") or None
