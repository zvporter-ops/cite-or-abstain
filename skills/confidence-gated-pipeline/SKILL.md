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

## The output gate

After generation, check the reply mechanically:

- Quoted spans must appear verbatim in the evidence.
- Figures, dates, URLs and identifiers must appear in a retrieved passage.
- Prose that merely paraphrases is **not** a violation. Flagging paraphrase makes the
  gate unusable.

## Anti-patterns

- Asking the model "are you sure?" — that returns another generated number.
- Treating retrieval as proof. Retrieval can return material *about* the topic that does
  not answer the question.
- Blocking on soft signals. Only narrow, verifiable claims should block.
