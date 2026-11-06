"""Corpus index: dependency-free BM25 over tiered roots.

Two tiers matter. ``internal`` is anything the user fed the deployment -- transcripts,
contracts, code, spreadsheets -- and it outranks everything else. ``public`` is reference
material anyone could have. The tier travels with every passage so the pipeline can
answer from internal first and label what it used.

The index is deliberately crude -- no embeddings, no model call, no third-party
dependency -- because the gate it feeds must be inspectable. A score you cannot explain
is a score you cannot trust.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import ingest

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".obsidian", ".trash",
    "Library", "Applications",
}
MAX_FILE_BYTES = 2_000_000
MAX_FILES = 4_000
MAX_PASSAGES_PER_DOC = 200
PASSAGE_CHARS = 1_200

INTERNAL = "internal"
PUBLIC = "public"
TIERS = (INTERNAL, PUBLIC)

_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019\-]{2,}")

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "who", "why", "how", "are", "was", "were", "been", "being", "have",
    "has", "had", "does", "did", "can", "could", "should", "would", "will", "you",
    "your", "our", "its", "their", "there", "then", "than", "into", "about", "not",
    "all", "any", "but", "out", "one", "two", "also", "such", "only", "more",
}


def content_words(text: str) -> list[str]:
    """Lowercased content words, in order, duplicates kept (tf needs them)."""
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP]


def related(left: str, right: str) -> bool:
    """True when two words are the same word form (prefix match, 4-char floor)."""
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False
    return left.startswith(right) or right.startswith(left)


def relevance(query: str, passage: "Passage") -> float:
    """Share of the query's content words this passage actually covers.

    A single incidental word ("window" in a launch memo) must not make a question look
    covered by internal material -- that would withhold public answers for no reason.
    """
    wanted = set(content_words(query))
    if not wanted:
        return 0.0
    have = content_words(passage.text)
    hits = sum(1 for word in wanted if any(related(word, term) for term in have))
    return hits / len(wanted)


@dataclass(frozen=True)
class Passage:
    """One retrievable unit: a document slice, where it came from, and its tier."""

    doc_id: int
    path: str
    start_line: int
    text: str
    score: float = 0.0
    tier: str = INTERNAL

    def cite(self) -> str:
        return f"{self.path}:{self.start_line}"

    def label(self) -> str:
        return f"{self.tier} {self.path}:{self.start_line}"


@dataclass
class IndexStats:
    documents: int = 0
    passages: int = 0
    terms: int = 0
    skipped: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)
    by_tier: dict[str, int] = field(default_factory=dict)


def normalize_roots(roots: Iterable) -> list[tuple[Path, str]]:
    """Accept ``"path"`` or ``{"path": ..., "tier": ...}``; default tier is internal."""
    normalized: list[tuple[Path, str]] = []
    for entry in roots:
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("root") or ""
            tier = str(entry.get("tier") or INTERNAL).lower()
        else:
            path, tier = str(entry), INTERNAL
        if tier not in TIERS:
            tier = INTERNAL
        if str(path).strip():
            normalized.append((Path(str(path)).expanduser(), tier))
    return normalized


class CorpusIndex:
    """SQLite-backed BM25 index over tiered text sources."""

    def __init__(self, roots: Iterable, db_path: str | Path,
                 max_files: int = MAX_FILES):
        self.sources = normalize_roots(roots)
        self.db_path = Path(db_path).expanduser()
        self.max_files = max_files
        self._conn: sqlite3.Connection | None = None
        self._terms: dict[str, int] = {}
        self._n_passages = 0
        self._avgdl = 1.0

    @property
    def roots(self) -> list[Path]:
        return [path for path, _ in self.sources]

    # ---------------------------------------------------------------- schema

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                passages INTEGER NOT NULL DEFAULT 0,
                tier TEXT NOT NULL DEFAULT 'internal'
            );
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                dl INTEGER NOT NULL,
                tier TEXT NOT NULL DEFAULT 'internal'
            );
            CREATE INDEX IF NOT EXISTS passages_doc ON passages(doc_id);
            CREATE TABLE IF NOT EXISTS terms (
                term TEXT NOT NULL,
                passage_id INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term, passage_id)
            );
            CREATE INDEX IF NOT EXISTS terms_term ON terms(term);
            CREATE TABLE IF NOT EXISTS term_df (
                term TEXT PRIMARY KEY,
                df INTEGER NOT NULL
            );
            """
        )
        self._migrate(conn)
        self._conn = conn
        self._load_stats()
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add the tier column to an index built before tiers existed."""
        for table in ("docs", "passages"):
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if "tier" not in columns:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN tier TEXT NOT NULL DEFAULT '{INTERNAL}'"
                )

    def _load_stats(self) -> None:
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*), COALESCE(AVG(dl), 1.0) FROM passages").fetchone()
        self._n_passages = int(row[0] or 0)
        self._avgdl = float(row[1] or 1.0)
        self._terms = {t: int(df) for t, df in conn.execute("SELECT term, df FROM term_df")}

    # ----------------------------------------------------------------- build

    def build(self, force: bool = False) -> IndexStats:
        """Index every changed file under the sources. Returns what happened."""
        conn = self._connect()
        stats = IndexStats()
        seen = 0
        keep: set[str] = set()
        for path, tier in self._walk():
            seen += 1
            if seen > self.max_files:
                break
            try:
                st = path.stat()
            except OSError as exc:
                stats.errors.append(f"{path}: {exc}")
                continue
            keep.add(str(path))
            if st.st_size > MAX_FILE_BYTES:
                stats.skipped += 1
                continue
            key = str(path)
            row = conn.execute(
                "SELECT id, mtime, size FROM docs WHERE path = ?", (key,)
            ).fetchone()
            if row and not force and abs(row[1] - st.st_mtime) < 1 and row[2] == st.st_size:
                conn.execute("UPDATE docs SET tier = ? WHERE id = ?", (tier, int(row[0])))
                stats.documents += 1
                continue
            text = ingest.extract_text(path, MAX_FILE_BYTES)
            if not text:
                stats.skipped += 1
                continue
            if row:
                self._drop_doc(int(row[0]))
                doc_id = int(row[0])
                conn.execute(
                    "UPDATE docs SET mtime = ?, size = ?, passages = ?, tier = ? WHERE id = ?",
                    (st.st_mtime, st.st_size, 0, tier, doc_id),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO docs(path, mtime, size, passages, tier) VALUES (?,?,?,0,?)",
                    (key, st.st_mtime, st.st_size, tier),
                )
                doc_id = int(cur.lastrowid or 0)
            written = self._index_document(conn, doc_id, tier, text)
            conn.execute("UPDATE docs SET passages = ? WHERE id = ?", (written, doc_id))
            stats.documents += 1
        if seen <= self.max_files:
            stats.removed = self._prune(conn, keep)
        conn.commit()
        self._load_stats()
        stats.passages = self._n_passages
        stats.terms = len(self._terms)
        stats.by_tier = self._tier_counts()
        return stats

    def _walk(self):
        for root, tier in self.sources:
            if root.is_file():
                yield root, tier
                continue
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for name in filenames:
                    if name.startswith("."):
                        continue
                    path = Path(dirpath) / name
                    if ingest.supported(path):
                        yield path, tier

    def _drop_doc(self, doc_id: int) -> None:
        conn = self._conn
        assert conn is not None
        ids = [int(r[0]) for r in conn.execute(
            "SELECT id FROM passages WHERE doc_id = ?", (doc_id,)
        )]
        for pid in ids:
            conn.execute("DELETE FROM terms WHERE passage_id = ?", (pid,))
        conn.execute("DELETE FROM passages WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM term_df")
        conn.execute(
            "INSERT INTO term_df(term, df) "
            "SELECT term, COUNT(DISTINCT passage_id) FROM terms GROUP BY term"
        )

    def _prune(self, conn: sqlite3.Connection, keep: set[str]) -> int:
        """Drop documents that no longer sit under any configured source.

        Without this, re-pointing ``corpus_roots`` would leave the old material
        answering questions -- including material the user just removed.
        """
        removed = 0
        for doc_id, path in conn.execute("SELECT id, path FROM docs").fetchall():
            if path in keep:
                continue
            self._drop_doc(int(doc_id))
            conn.execute("DELETE FROM docs WHERE id = ?", (int(doc_id),))
            removed += 1
        return removed

    def _index_document(self, conn: sqlite3.Connection, doc_id: int,
                        tier: str, text: str) -> int:
        written = 0
        for start_line, chunk in _chunk(text):
            words = content_words(chunk)
            if not words:
                continue
            cur = conn.execute(
                "INSERT INTO passages(doc_id, start_line, text, dl, tier) VALUES (?,?,?,?,?)",
                (doc_id, start_line, chunk, len(words), tier),
            )
            pid = int(cur.lastrowid or 0)
            counts: dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            conn.executemany(
                "INSERT OR REPLACE INTO terms(term, passage_id, tf) VALUES (?,?,?)",
                [(t, pid, tf) for t, tf in counts.items()],
            )
            # df counts passages, not occurrences: one increment per passage.
            conn.executemany(
                "INSERT INTO term_df(term, df) VALUES (?, 1) "
                "ON CONFLICT(term) DO UPDATE SET df = df + 1",
                [(t,) for t in counts],
            )
            written += 1
            if written >= MAX_PASSAGES_PER_DOC:
                break
        return written

    # -------------------------------------------------------------- retrieve

    def _expand(self, word: str) -> list[str]:
        """Indexed terms that are the same word form as ``word``."""
        if not self._terms:
            return []
        return [term for term in self._terms if related(word, term)]

    def retrieve(self, query: str, k: int = 4, tier: str | None = None) -> list[Passage]:
        conn = self._connect()
        if self._n_passages == 0:
            return []
        words = content_words(query)
        if not words:
            return []
        wanted: dict[str, float] = {}
        for word in words:
            for term in self._expand(word):
                wanted[term] = self._idf(term)
        if not wanted:
            return []
        marks = ",".join("?" for _ in wanted)
        params: list = list(wanted)
        tier_clause = ""
        if tier:
            tier_clause = " AND p.tier = ?"
            params.append(tier)
        rows = conn.execute(
            f"SELECT p.id, p.doc_id, d.path, p.start_line, p.text, p.dl, p.tier, t.term, t.tf "
            f"FROM terms t "
            f"JOIN passages p ON p.id = t.passage_id "
            f"JOIN docs d ON d.id = p.doc_id "
            f"WHERE t.term IN ({marks}){tier_clause}",
            tuple(params),
        ).fetchall()
        scored: dict[int, tuple[float, tuple]] = {}
        k1, b = 1.5, 0.75
        for pid, doc_id, path, start_line, text, dl, p_tier, term, tf in rows:
            idf = wanted[term]
            denom = tf + k1 * (1 - b + b * (dl / self._avgdl))
            score = idf * (tf * (k1 + 1)) / denom
            prev = scored.get(pid)
            if prev is None:
                scored[pid] = (score, (doc_id, path, start_line, text, p_tier))
            else:
                scored[pid] = (prev[0] + score, prev[1])
        ranked = sorted(scored.values(), key=lambda item: item[0], reverse=True)[:k]
        return [
            Passage(doc_id=d, path=p, start_line=s, text=t, score=round(sc, 4), tier=ti)
            for sc, (d, p, s, t, ti) in ranked
        ]

    def sample_passages(self, tier: str = INTERNAL, n: int = 3) -> list[Passage]:
        """A few passages from ``tier``, for audits and scorecards."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT p.doc_id, d.path, p.start_line, p.text, p.tier "
            "FROM passages p JOIN docs d ON d.id = p.doc_id "
            "WHERE p.tier = ? ORDER BY p.id LIMIT ?",
            (tier, n),
        ).fetchall()
        return [
            Passage(doc_id=d, path=path, start_line=s, text=text, tier=ti)
            for d, path, s, text, ti in rows
        ]

    def _idf(self, term: str) -> float:
        import math
        df = self._terms.get(term, 0)
        n = max(self._n_passages, 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _tier_counts(self) -> dict[str, int]:
        conn = self._connect()
        return {
            tier: int(n) for tier, n in conn.execute(
                "SELECT tier, COUNT(*) FROM passages GROUP BY tier"
            )
        }

    def stats(self) -> dict:
        conn = self._connect()
        docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        return {
            "documents": int(docs),
            "passages": self._n_passages,
            "terms": len(self._terms),
            "by_tier": self._tier_counts(),
            "sources": [{"path": str(p), "tier": t} for p, t in self.sources],
            "db": str(self.db_path),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _chunk(text: str) -> list[tuple[int, str]]:
    """Split a document into paragraph-aligned chunks of ~PASSAGE_CHARS."""
    lines = text.splitlines()
    chunks: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 1
    size = 0
    for idx, line in enumerate(lines, start=1):
        if not buf:
            start = idx
        buf.append(line)
        size += len(line) + 1
        if size >= PASSAGE_CHARS and not line.strip():
            chunks.append((start, "\n".join(buf).strip()))
            buf, size = [], 0
    if buf:
        tail = "\n".join(buf).strip()
        if tail:
            chunks.append((start, tail))
    return [(s, c) for s, c in chunks if c]
