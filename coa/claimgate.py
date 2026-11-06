"""The mechanical claim gate, with provenance.

The second half of the thesis: the input gate decides whether an answer may exist;
this gate decides whether what was said is actually in the evidence, and *which tier*
that evidence came from. It never asks a model for an opinion -- a model asked to check
its own answer defends it.

Three outcomes, deliberately different:

``violation``   a quote, figure, date, URL or identifier found in neither the internal
                nor the public evidence. Narrow by construction, so a miss is strong
                evidence of fabrication rather than paraphrase.
``public``      a claim that rests on public material only, when public answers are
                permitted. Disclosed, never blocked: it is legitimate, but the user has
                to know it did not come from their own documents.
``soft``        prose with weak overlap, or a proper-noun run with no direct support.
                Reported, never blocking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .confidence import lexical_overlap
from .corpus import INTERNAL, PUBLIC, Passage

QUOTE_PATTERNS = (
    re.compile(r'"([^"]{8,})"'),
    re.compile(r"\u201c([^\u201d]{8,})\u201d"),
)

FIGURE = re.compile(
    r"(?<![\w.])("
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # 1,200 / 12,000.5
    r"|\d+\.\d+"                     # 3.14
    r"|\d+\s?%"                      # 47%
    r"|\d{2,}"                       # 42, 2026
    r")(?![\w])"
)
DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4})\b"
)
URL = re.compile(r"https?://[^\s)>\]]+")
IDENTIFIER = re.compile(r"`([^`]{2,60})`")
PROPER_RUN = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b")
SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
SUPPORT_OVERLAP = 0.2

# Line and passage references are structure, not claims. Masking them keeps the figure
# check honest without flagging every citation ("lines 5 and 11-12", "(12 lines)").
CITATION_NUMBERS = (
    re.compile(r"\blines?\s+\d+(?:\s*(?:and|,|-|\u2013)\s*\d+)*", re.IGNORECASE),
    re.compile(r"\(\s*\d+\s+lines?\s*\)", re.IGNORECASE),
    re.compile(r"\bpassages?\s+\[?\d+[^\s]*", re.IGNORECASE),
)


def _norm(text: str) -> str:
    """Casefold, unify quotes, collapse whitespace -- the comparison form."""
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass
class Finding:
    kind: str            # quote | figure | date | url | identifier | name | prose
    severity: str        # violation | public | soft
    span: str
    detail: str
    provenance: str = "none"   # internal | public | none


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    supported: list[tuple[str, str, str]] = field(default_factory=list)
    coverage_internal: float = 0.0
    coverage_any: float = 0.0
    passage_count: int = 0
    internal_count: int = 0
    public_count: int = 0

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "violation"]

    @property
    def public_only(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "public"]

    @property
    def coverage(self) -> float:
        """Internal-tier sentence support; the headline number."""
        return self.coverage_internal

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "coverage_internal": round(self.coverage_internal, 3),
            "coverage_any": round(self.coverage_any, 3),
            "passage_count": self.passage_count,
            "internal_count": self.internal_count,
            "public_count": self.public_count,
            "violations": [f.__dict__ for f in self.violations],
            "public_only": [f.__dict__ for f in self.public_only],
            "soft": [f.__dict__ for f in self.findings if f.severity == "soft"],
            "supported": [
                {"sentence": s, "source": c, "tier": t} for s, c, t in self.supported
            ],
        }


def _tier_of(needle: str, internal_text: str, all_text: str) -> str:
    """Which tier contains ``needle``: internal, public, or none."""
    normalized = _norm(needle)
    if internal_text and normalized in internal_text:
        return INTERNAL
    if all_text and normalized in all_text:
        return PUBLIC
    return "none"


def check(answer: str, passages: list[Passage], allow_public: bool = False) -> Report:
    """Gate ``answer`` against ``passages``.

    ``allow_public`` permits claims that rest on public material: they are disclosed
    rather than blocked. It is set when the internal tier was silent for the question,
    because refusing every public answer would make the plugin useless on day one.
    """
    internal = [p for p in passages if p.tier == INTERNAL]
    internal_text = "\n\n".join(_norm(p.text) for p in internal)
    all_text = "\n\n".join(_norm(p.text) for p in passages)
    report = Report(
        passage_count=len(passages),
        internal_count=len(internal),
        public_count=len(passages) - len(internal),
    )

    def record(kind: str, span: str) -> None:
        tier = _tier_of(span, internal_text, all_text)
        if tier == INTERNAL:
            return  # supported by the user's own material
        if tier == PUBLIC:
            # Legitimate evidence, but not the user's. Disclose it when public answers
            # are allowed; treat it as unsupported when they are not.
            report.findings.append(Finding(
                kind=kind,
                severity="public" if allow_public else "violation",
                span=span, provenance=PUBLIC,
                detail=("appears only in public material, not in the internal corpus"
                        if allow_public else
                        "found only in public material, which this answer may not use")))
            return
        report.findings.append(Finding(
            kind=kind, severity="violation", span=span, provenance="none",
            detail=f"{kind} found in no retrieved material"))

    # 1. Quoted spans must be verbatim in some evidence tier.
    for pattern in QUOTE_PATTERNS:
        for match in pattern.finditer(answer):
            record("quote", match.group(1).strip())

    # 2. Figures, dates, URLs and identifiers must appear in evidence. Figures are read
    # from a copy with citation numbers masked out, so "lines 11-12" is not a claim.
    figures_from = answer
    for pattern in CITATION_NUMBERS:
        figures_from = pattern.sub(" ", figures_from)
    for kind, pattern, source in (("figure", FIGURE, figures_from), ("date", DATE, answer),
                                  ("url", URL, answer), ("identifier", IDENTIFIER, answer)):
        for match in pattern.finditer(source):
            span = (match.group(1) if match.groups() else match.group(0)).strip()
            record(kind, span)

    # 3. Proper-noun runs are reported, never blocking.
    for match in PROPER_RUN.finditer(answer):
        span = match.group(1)
        if _tier_of(span, internal_text, all_text) == "none":
            report.findings.append(Finding(
                kind="name", severity="soft", span=span,
                detail="proper-noun run has no direct support"))

    # 4. Sentence-level support, attributed to a tier.
    sentences = [s.strip() for s in SENTENCE.findall(answer) if s.strip()]
    internal_hits = public_hits = 0
    for sentence in sentences:
        if not passages:
            break
        best_internal = max(
            internal, key=lambda p: lexical_overlap(sentence, p.text), default=None)
        if best_internal and lexical_overlap(sentence, best_internal.text) >= SUPPORT_OVERLAP:
            internal_hits += 1
            report.supported.append((sentence, best_internal.cite(), INTERNAL))
            continue
        best_any = max(passages, key=lambda p: lexical_overlap(sentence, p.text))
        overlap = lexical_overlap(sentence, best_any.text)
        if overlap >= SUPPORT_OVERLAP:
            public_hits += 1
            report.supported.append((sentence, best_any.cite(), best_any.tier))
        else:
            report.findings.append(Finding(
                kind="prose", severity="soft", span=sentence[:120],
                detail=f"weak support (best overlap {overlap:.2f})"))

    total = len(sentences) or 1
    report.coverage_internal = internal_hits / total
    report.coverage_any = (internal_hits + public_hits) / total
    return report


def render_report(report: Report) -> str:
    """Compact markdown block appended to a reply in ``flag`` mode."""
    lines = ["", "---", "**Evidence gate**"]
    if report.violations:
        lines.append(f"**{len(report.violations)} unsupported claim(s):**")
        for finding in report.violations:
            lines.append(f"- {finding.kind}: `{finding.span[:80]}` — {finding.detail}")
    else:
        lines.append("No unsupported claims.")
    if report.public_only:
        lines.append(
            f"**{len(report.public_only)} claim(s) rest on public material, not your "
            "internal corpus:**")
        for finding in report.public_only:
            lines.append(f"- {finding.kind}: `{finding.span[:80]}`")
    lines.append(
        f"Support: {report.coverage_internal:.0%} from internal material, "
        f"{report.coverage_any:.0%} from any evidence "
        f"({report.internal_count} internal / {report.public_count} public passages).")
    soft = [f for f in report.findings if f.severity == "soft"]
    if soft:
        lines.append(f"Soft flags: {len(soft)} (paraphrase or unsupported names; not blocking).")
    return "\n".join(lines)
