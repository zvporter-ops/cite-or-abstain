# Cite or Abstain

**An evidence gate for any AI deployment, on day one.**

Point Hermes at a folder of your own material and every answer either comes from it or is
refused. Point it at nothing and it says so instead of inventing an answer. Add public
reference material and answers that lean on it are labelled as public, not as yours.
Nothing is ever graded by the model that wrote it.

```
You:  What is the Project Falcon budget and launch window?
It:   From your internal memo: $250,000 approved plus a 2% contingency, launch window
      March 14–28 2026, vendor Helios Systems...
      ---
      Evidence gate
      2 unsupported claims: figure 5,000 / 255,000 — not in any retrieved passage
      Support: 94% from internal material
```

The model did arithmetic nobody asked for. The gate caught it, because arithmetic is not
evidence. You decide whether that is fine; in `strict` mode it never reaches you.

## Why

A local model is easy to install and hard to trust. It answers confidently about things
that are not in your documents, cites files it has never read, and produces figures that
look plausible. Asking the model to check itself does not work — a model asked to police
its own answer defends it. So the check has to live outside the model, and it has to be
mechanical enough that you can read the code and know what it does.

## The hierarchy: internal > public > refuse

Every passage carries a tier.

| Tier | What it is | How answers use it |
|---|---|---|
| **internal** | Anything you feed the deployment: transcripts, contracts, code, spreadsheets, PDFs, emails | Answered from first, cited first |
| **public** | Reference material anyone could have | Used only when internal material does not cover the question, and labelled as public |
| **none** | Nothing retrieved | The gate flags the claim; `strict` refuses |

The point is the transition. On day one there is no internal material, so the system
answers from public reference and tells you it did. The moment you add a contract, a memo
or a transcript that covers the question, that material takes precedence and the answer
stops being a public one.

## How it works

Two gates and one clearance rule, all mechanical.

**Input gate** (`pre_llm_call`) — retrieves internal passages first, then public, filters
out passages that only share an incidental word with the question, and injects the
surviving evidence with its tier. When nothing covers the question it injects a silence
instruction instead of letting the model free-associate.

**Output gate** (`transform_llm_output`) — checks the finished reply against the passages
that turn actually retrieved:

- quoted spans must be verbatim in some evidence tier;
- figures, dates, URLs and identifiers must appear in a retrieved passage (line and
  passage references are masked, so citations are not mistaken for claims);
- claims resting only on public material are disclosed as such;
- prose with weak support is reported as a soft flag, never blocking.

**Clearance rule** — if internal material matches the question and the active model is not
running locally, the material is withheld and the model is told to say so. A cloud model
never receives your private text. Local means loopback, RFC1918, Tailscale CGNAT
(`100.64.0.0/10`) or a `.local`/`.lan`/`.internal` name; anything unclassified counts as
remote, because failing closed is the only safe default.

## Trust scorecard

```
/trust
```

Runs a battery against the deployment you actually configured and scores it:

- does the gate catch an invented quote and an invented figure?
- does it leave legitimate paraphrase alone?
- does the pipeline abstain on questions your corpus cannot answer?
- how much internal material is there, and would it be shared with this model?

The gate probes are deterministic; the pipeline probes are real model calls. A score is a
measurement of this deployment, not a claim about the plugin in general.

## Install

```bash
hermes plugins install zacharyporter/cite-or-abstain --enable
```

Then point it at your material:

```yaml
plugins:
  entries:
    cite-or-abstain:
      settings:
        corpus_roots:      # internal, outranks everything
          - ~/Documents/contracts
          - {path: ~/notes/transcripts, tier: internal}
        public_roots:      # public reference, labelled when used
          - ~/reference/rfcs
        strictness: flag   # off | flag | strict
```

Reads Markdown, text, reStructuredText, HTML, PDF, Word, Excel, PowerPoint and images
(images only when `tesseract` is installed). Files with a binary header, archives hiding
binaries and unsupported formats are skipped rather than guessed at.

`/ground off|flag|strict` changes the mode mid-session.

## Tools

| Tool | What it does |
|---|---|
| `corpus_ask` | Answer from the corpus through the three-phase pipeline, internal tier first |
| `evidence_report` | Check a reply against this turn's evidence |
| `corpus_audit` | Score the deployment (the `/trust` battery, as JSON) |

## Limits, stated plainly

- **Arithmetic is flagged.** A derived figure (`$250,000 × 2%`) is not in the corpus, so
  it is reported as unsupported. That is deliberate: the gate verifies what was said
  against what was read, not whether the maths is right.
- **Retrieval is BM25, not embeddings.** No dependency, no model call, and a score you can
  explain. It will miss paraphrases that share no vocabulary.
- **The gate checks form, not truth.** It catches invented quotes, figures and citations.
  A claim that restates the evidence wrongly can still pass.
- **Streaming is held.** The reply is withheld until the gate has run, so you never see
  text that has not been checked. That costs token-by-token streaming while the plugin is
  enabled.
- **It gates the main turn.** Tool subcalls and delegated agents are not covered.

## Privacy and side effects

- Reads only the directories you configure, plus its own index in
  `$HERMES_HOME/plugin-data/cite-or-abstain/`.
- Makes model calls only through Hermes's configured model, or the OpenAI-compatible
  `endpoint` you set yourself.
- No telemetry, no network calls beyond those, no self-update, no credential files.
- `COA_API_KEY` (optional) is read through Hermes's secret scope, never from a file.

## Related work

The two halves come from earlier projects: [`cgp`](https://github.com/zacharyporter/cgp)
(the confidence-gated input loop) and `transcript-forge` (the output gate). This plugin is
where they meet, with tiers and a clearance rule added.

MIT licensed.
