"""Domain lenses for the three phases.

Vendored from ``cgp.domains``. A lens changes what "enough context" means, not the
mechanics: the same three-phase structure runs for research, code review, debugging,
system ops or content work; only the evaluation criteria differ.
"""

from __future__ import annotations

LENSES = {
    "research": {
        "role": "a research analyst deciding whether the provided sources can answer a question",
        "phase2_focus": "the specific sources, definitions, or measurements that would settle the question",
        "phase3_focus": "an answer grounded in the provided sources, with explicit gaps",
    },
    "code_review": {
        "role": "a reviewer deciding whether the provided code and tests are enough to judge a change",
        "phase2_focus": "the call sites, tests, schema, and invariants that the change touches",
        "phase3_focus": "a review verdict grounded in the provided code, with what remains unverified",
    },
    "debugging": {
        "role": "a diagnostician deciding whether the provided logs and code explain a failure",
        "phase2_focus": "the log lines, stack frames, and configuration values needed to confirm a cause",
        "phase3_focus": "the most likely cause with the evidence for it, and what would confirm or rule it out",
    },
    "system_ops": {
        "role": "an operator deciding whether the provided state and history make a change safe",
        "phase2_focus": "current state, recent changes, dependencies, and rollback paths",
        "phase3_focus": "a go/no-go recommendation with the risks the evidence does not cover",
    },
    "content_creation": {
        "role": "an editor deciding whether the provided material supports an angle",
        "phase2_focus": "existing coverage, audience assumptions, and source material for the claims",
        "phase3_focus": "a draft direction grounded in the provided material, with unsupported claims named",
    },
}

DOMAINS = tuple(LENSES)


def phase1_prompt(domain: str) -> str:
    lens = LENSES[domain]
    return f"""You are {lens['role']}.

Evaluate whether the provided context is sufficient to answer the question. Do not answer it.

Output only this structure:

CONFIDENCE: [high|medium|low]
REASONS:
- one reason what context you have
- one reason what is missing or unclear
- one reason what additional material would help

Do not hedge. Do not add text outside this structure."""


def phase2_prompt(domain: str) -> str:
    lens = LENSES[domain]
    return f"""You are {lens['role']}.

The previous phase found the context insufficient. Name what you need to look up.

Output only this structure:

CONTEXT_NEEDED:
- a short search phrase for {lens['phase2_focus']}
- another search phrase
- another search phrase

Each line is a search target for the knowledge base, not a question for the user.
Keep each under fifteen words."""


def phase3_prompt(domain: str) -> str:
    lens = LENSES[domain]
    return f"""You are {lens['role']}.

Answer the question using only the context provided, plus the confidence assessment and any
additional context retrieved. Produce {lens['phase3_focus']}.

Output only this structure:

ANSWER:
the answer

LIMITATIONS:
- what the provided context could not establish

RECOMMENDATIONS:
- what additional material would improve the answer

Rules: never attribute a fact, number, or claim that is not in the provided context. If the context
does not support an answer, say so in ANSWER and list why under LIMITATIONS."""


def build_prompts(domain: str) -> dict[str, str]:
    if domain not in LENSES:
        raise KeyError(f"unknown domain {domain!r}; choose from {sorted(LENSES)}")
    return {
        "phase1": phase1_prompt(domain),
        "phase2": phase2_prompt(domain),
        "phase3": phase3_prompt(domain),
    }
