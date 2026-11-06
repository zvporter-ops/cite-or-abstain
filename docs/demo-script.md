# Demo script — 90 seconds

Screen-record a terminal at a readable font size (16pt+), plus one shot of the Desktop pane.
Everything below is real output from this repo's `examples/` corpus; nothing is staged.

Set up before recording (not on camera):

```bash
cd /Users/zacharyporter/Documents/AI_Dev/github-repos/cite-or-abstain
hermes config set plugins.entries.cite-or-abstain.settings.corpus_roots '[]'   # start empty
hermes plugins disable cite-or-abstain && hermes plugins enable cite-or-abstain
```

## 0:00–0:12 — the problem, in one breath

> "A local model is easy to install and hard to trust. It answers confidently about things
> that aren't in your documents, and cites files it never read. Asking it to check itself
> doesn't work — a model asked to police its own answer defends it."

## 0:12–0:25 — day one: nothing loaded, so it says so

```bash
hermes -z "What is the Project Falcon budget and launch window?"
```

Expect: the corpus is silent, no figures invented, an evidence gate block reporting 0%
support. *(Beat: it refuses instead of bluffing.)*

## 0:25–0:48 — drop in your own material; it takes precedence

```bash
hermes config set plugins.entries.cite-or-abstain.settings.corpus_roots \
  '[{"path":"examples/internal","tier":"internal"}]'
hermes config set plugins.entries.cite-or-abstain.settings.public_roots \
  '["examples/public"]'
hermes plugins disable cite-or-abstain && hermes plugins enable cite-or-abstain
hermes -z "What is the Project Falcon budget and launch window?"
```

Expect: answered from `examples/internal/falcon-memo.md`, cited as internal, gate clean.
*(Beat: the same question now has an answer, and it comes from your material.)*

## 0:48–1:00 — public material is labelled, not laundered

```bash
hermes -z "What is a rollback window?"
```

Expect: "rests on public material only — internal corpus does not define the term",
`0% from internal material`. *(Beat: it knows the difference between the open web and your
documents.)*

## 1:00–1:15 — the gate catches what the model added

```bash
hermes -z "What is the Project Falcon budget?"
```

Expect: the model computes the 2% contingency ($5,000, $255,000) and the gate flags both as
figures that appear in no retrieved passage. *(Beat: arithmetic is not evidence.)*

```bash
hermes -z "/ground strict"
hermes -z "What is the Project Falcon budget?"
```

Expect: a refusal naming the kind of claim, never echoing the fabricated number.

## 1:15–1:30 — the score

```bash
hermes -z "/trust"
```

Expect: `Trust report — trusted (100/100)`, four gate probes passing, 2/2 abstentions,
1/1 answered, clearance line showing the model is local.

Cut to the Desktop **evidence** pane: corpus tiers, last gate outcome, mode switch, audit
button. End on the status-bar chip.

## Caption

> Cite or Abstain — an evidence gate for any AI deployment. Internal material outranks
> public, public answers are labelled, and nothing unsupported reaches you.

Tag @NousResearch. Under 90 seconds.
