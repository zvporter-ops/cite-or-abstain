# Coldstart

**An evidence gate that works from cold start.**

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

The name is the state it is built for. A cold start is a deployment with nothing of yours
behind it yet — and that is exactly where a model is most willing to invent. Coldstart
makes that state visible instead of hiding it, then shows you the moment it stops being
true.

## Standing rules: a correction is a defect

The document gate answers *what does it know*. This answers *how does it behave*, and it
starts empty in the same way.

A correction you have to repeat is not a personality quirk — it is a save that did not
take. So Coldstart records corrections:

```
You:  no, use plain language and drop the hedging footer
It:   Understood.
      ---
      Rule 1 saved: use plain language and drop the hedging footer
```

State the same thing again and the rule becomes **hard**, because the first save failed:

```
      ---
      Rule 1 is hard now: use plain language and drop the hedging footer
      It had been stated before, so the earlier save did not take.
      It is injected into every turn from here on.
```

Hard rules are injected at the start of every turn, under a character budget
(`rules_max_chars`), ahead of any retrieved evidence. Detection is deliberately
conservative: a question, a bare "thanks", or an ordinary statement never becomes a rule,
because a false positive injects an instruction the user never gave.

`/rules` lists what has been captured, `/rules forget <id>` drops one, `/rules hard <id>`
promotes one. `rules_enabled: false` turns the ledger off and leaves only the document gate.

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

The rule covers every path that touches your material, not just the turn: `corpus_ask`
answers from public material only when the model it would call is not local, `corpus_audit`
and `/trust` refuse rather than sample internal passages into a remote model, and the
clearance notice only points at `corpus_ask` when a local `endpoint` exists to answer
through. With no `endpoint` set, the plugin calls your session model, so a cloud session
gets public answers and an honest refusal — never your documents.

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
hermes plugins install zvporter-ops/coldstart --enable
```

Then point it at your material:

```yaml
plugins:
  entries:
    coldstart:
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

## Desktop pane

The same repo ships both halves. Hermes Desktop gets an **evidence** pane and a status-bar
chip, backed by `dashboard/plugin_api.py` at `/api/plugins/coldstart/`:

- corpus composition by tier, and whether the current model is local or remote;
- the last gated reply — clean, or the unsupported claims by kind, plus internal coverage;
- a mode switch (`off` / `flag` / `strict`) that writes the plugin setting;
- **Run audit**, which runs the same battery as `/trust` and shows the score.

The pane reads the same `coa.audit` module the command runs, so the number in the UI and
the number in the terminal cannot drift apart.

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
  `$HERMES_HOME/plugin-data/coldstart/`.
- Makes model calls only through Hermes's configured model, or the OpenAI-compatible
  `endpoint` you set yourself.
- No telemetry, no network calls beyond those, no self-update, no credential files.
- `COA_API_KEY` (optional) is read through Hermes's secret scope, never from a file.

## Related work

Seven small tools preceded this one, and each proved a property the gate now depends on:
`cgp` (separating evaluation from generation, so a model can admit it does not know),
`transcript-forge` (a mechanical quote gate over an artifact the model did not write),
`research-loop` (local material before the web), `context-pack` (evidence is not optional),
`claim-synth` (deterministic routing with no model in the loop), `vault-pipeline` (integrity
gates that only fire on new drift), and `hermes-ops` (report the one thing that is wrong).
They are local, unpublished repositories.

[`docs/ORIGINS.md`](docs/ORIGINS.md) records what each one established, what carried over, and
what did not — so the lineage is auditable rather than asserted.

MIT licensed.
