---
name: confidence-gated-pipeline
description: "Answer only over evidence: rate confidence, name the gap, retrieve it, answer, or abstain."
---

# Confidence-gated answering

Use this when a question must be answered from a specific corpus and a fluent wrong
answer is worse than no answer.

## The rule

Separate **evaluation** from **generation**. A model asked to answer will answer. A
model asked to judge whether the context is sufficient has no reason to invent one.

## Procedure

1. **Retrieve first.** Query the corpus before forming an opinion about it.
2. **Phase 1 — rate, don't answer.** Ask for `CONFIDENCE: [high|medium|low]` plus three
   reasons: what the context contains, what is missing, what would help. Do not answer
   the question in this call.
3. **Cross-check the rating.** A self-reported `high` over thin retrieval is the failure
   mode. Compare the query's vocabulary against the retrieved text. Demote `high` when
   coverage is low; promote `low` only when coverage is high.
4. **Phase 2 — turn the admission into a search.** Ask what to look up, retrieve those
   targets, and **re-score on everything retrieved**. Scoring only the first retrieval
   makes the loop decorative.
5. **Phase 3 — answer over evidence.** Never attribute a fact, number, or claim that is
   not in the provided context.
6. **At `low`, do not generate.** Return the refusal, the model's reasons as limitations,
   and the retrieval targets as recommendations.

## Tiers

Every passage carries a tier, and the order is the point:

1. **internal** — anything the user fed the deployment. Answer from here first, cite it first.
2. **public** — reference material anyone could have. Use it only when internal material does
   not cover the question, and say plainly that the answer rests on public sources.
3. **nothing** — refuse, and name the material that would answer it.

A question counts as covered by internal material only when the passages cover most of its
content words. One shared word ("window" in a launch memo) is not coverage — treating it as
such withholds legitimate public answers.

Private material never travels to a model that is not running locally. When internal
material matches and the model is remote, withhold it and say so.

## The output gate

After generation, check the reply mechanically:

- Quoted spans must appear verbatim in the evidence.
- Figures, dates, URLs and identifiers must appear in a retrieved passage. Mask line and
  passage references first, or every citation reads as a fabricated figure.
- A claim found only in public material is disclosed, not blocked — unless public answers
  were not permitted, in which case it is a violation.
- Prose that merely paraphrases is **not** a violation. Flagging paraphrase makes the
  gate unusable.

Do not echo the offending span in a refusal: repeating a fabricated figure hands it to the
user anyway. Name the kind of claim, not its content.

## Anti-patterns

- Asking the model "are you sure?" — that returns another generated number.
- Treating retrieval as proof. Retrieval can return material *about* the topic that does
  not answer the question.
- Blocking on soft signals. Only narrow, verifiable claims should block.
- Treating a derived figure as supported because its inputs were. Arithmetic is not
  evidence; flag it.
