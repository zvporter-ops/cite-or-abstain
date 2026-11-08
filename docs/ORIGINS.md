# Where this plugin comes from

Cite or Abstain is not a greenfield idea. It is the meeting point of seven small, stdlib-only
tools built over the preceding months, each of which proved one mechanical property that this
plugin now relies on. They are local repositories, unpublished, so they are named rather than
linked — the claims below come from their own READMEs, not from memory.

The two failures being addressed:

1. A small local model asked something it cannot answer answers anyway, fluently.
2. A fluent document quietly invents a number, a decision, or a quotation — and reads well
   enough that nobody checks.

## What each ancestor proved, and what carried over

| Repo | What it established | What this plugin took | What changed |
|---|---|---|---|
| `cgp` | Separating *evaluation* from *generation* lets a model admit the context is insufficient; at low confidence Phase 3 is never called and the run returns `abstained=true`. Confidence is adjusted by a signal that does not come from the model (lexical overlap). | Abstain when nothing covers the question; a relevance floor on retrieval; the answer is generated only over what was retrieved. | The gate is no longer a separate three-phase pipeline the user invokes — it runs on every Hermes turn, as a hook. |
| `transcript-forge` | Two independent reads beat self-review: a model handed a document it did not write audits it like a reviewer, while a model asked to check its own answer defends it. The quote gate is mechanical, not model-based. | The output gate checks the finished reply against the passages that turn actually retrieved; quoted spans must be verbatim. | Extended from quotes to figures, dates, URLs and identifiers; and from one artifact to every reply. |
| `research-loop` | The vault is searched before the web. Word-boundary matching, because the script it replaced matched `art` inside `partial`. | Internal material is retrieved first and outranks public material; relevance is word-bounded. | Web search is gone; the public tier is material you point at, and answers leaning on it are labelled rather than blended in. |
| `context-pack` | Evidence is not optional. If the mandatory material does not fit the budget, the honest answer is to say so, not to silently trim it. | When nothing is retrieved, the reply is refused and the corpus is named as silent — rather than answering from a thinner context. | The budget is a character cap on rendered evidence; the manifest of what was dropped is now the tier disclosure. |
| `claim-synth` | Claims can be routed deterministically — speaker filter, word bounds, most-specific-wins — with no model in the loop, idempotently. | No model call for retrieval or for claim detection. The gate is inspectable and its verdict is reproducible. | Routing to wiki pages became claim checking against retrieved passages. |
| `vault-pipeline` | Integrity gates with a baseline: only *new* drift fires, and links are never invented when nothing is shared. | The scorecard reports what the gate caught, not what it might catch; a clean reply is reported as clean rather than as a warning. | A baseline file became a per-turn gate plus a four-probe battery (`/trust`, `corpus_audit`). |
| `hermes-ops` | Operational scripts must report the one thing that is wrong, and exit 0 — the text is the signal. | The evidence report names claim kinds and the passages they should have come from, and does not block unless strictness says so. | Alerts became annotations on the reply itself. |

## The design frame

Two notes in the vault set the conceptual target, and this plugin is the mechanical version of
them:

- `Vault_01/wiki/concepts/AI-Agent-Architecture-2026.md` — the verifier-critic pattern, and the
  failure it has: a verifier that keeps finding problems never terminates, and a model asked to
  critique its own output tends to defend it. This plugin replaces the critic with a check that
  cannot be argued with.
- `Vault_01/wiki/concepts/LLM-Alignment-Safety-Evaluation-2026.md` — over-refusal and
  under-refusal are measured in different places and rarely connected. The three modes
  (`off` / `flag` / `strict`) are that trade-off made explicit and switchable, rather than
  guessed at.

## What did not carry over

Stated so nobody assumes more coverage than exists:

- Tool subcalls and delegated agents are not gated; only the main turn is.
- Arithmetic is flagged. A derived figure is not in the corpus, so the gate reports it.
- Retrieval is BM25, so a paraphrase sharing no vocabulary with the source is not found.
- The gate checks form, not truth: an invented quote or figure is caught, a misrestatement of
  real evidence can pass.
