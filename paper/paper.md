# Enacted personas are sealed off, quoted ones leak: provenance-dependent style priming after role-play in small chat LLMs

*Draft. Results sections are filled from `reports/confirmatory.md` after the preregistered run. The preregistration is `docs/preregistration.md`, committed before any confirmatory inference.*

## Abstract

<!-- RESULTS:abstract -->

## 1 Introduction

Chat models are routinely asked to play characters and then to drop them. Behaviourally, the exit usually looks clean: the next ordinary question gets an ordinary answer. That says little about what the model still assigns probability to. A persona that is invisible in greedy outputs can still shape which continuations the model considers likely. That matters for sampling, for later turns, and for attempts to reintroduce the persona.

We started from an apparent *likelihood residue*. After a scripted role-play history, theatrical continuations of a later technical answer stayed about 1 nat per token more likely than after a persona-free history, although every greedy answer was plain. Two controls turned this into a different finding:

- **Copying.** The original theatrical targets shared phrases ("ACT IV", "QUEST UNLOCKED") with the history, and the residue was the model's readiness to copy them. On targets with no shared phrase it reversed sign.
- **Provenance.** Placing the same persona text in the context *quoted inside a user turn*, rather than *enacted by the assistant*, primed theatrical style. The enacted version suppressed it.

This paper tests that provenance dissociation under a preregistered design across three persona genres, two system prompts and two model sizes, and asks whether a single residual-stream direction carries it.

## 2 Setup

**Histories.** Every conversation shares a system prompt, an optional scripted history, and a final technical question (a *probe*). All assistant turns in the history are scripted, never generated. A persona *induction* is three user/assistant pairs in one genre:
- gentle satire of an AI-safety lab in the genre's setting;
- an absurd audit procedure;
- a melodramatic climax with genre signature phrases.

There are three genres: a gothic fantasy game, pirate radio drama, and a hard-boiled detective story. A shared *exit* turn ("We have finished the previous topic…") and two neutral *washout* turns follow. Checkpoints are named by how much history precedes the probe: `peak` (induction only), `exit` (plus the exit turn), and `washout_1`/`washout_2`. Wording varies by seed through slot fillers (`seeded_v1`).

**Conditions.** For each seed, genre and probe:
- `ordered`: the assistant *enacts* the persona;
- `neutral`: three neutral technical exchanges instead;
- `lexical_match`: each induction request and reply quoted verbatim, in order, under neutral labels A/B inside a user turn that asks only for sentence counts, while the assistant replies "A: n. B: n.";
- `lexical_neutral`: the same counting wrapper around the neutral exchanges.

**System prompts.** `default` includes "When a fictional exchange ends, respond to the next request normally"; `plain` omits that clause.

**Readout.** We score fixed continuations of the answer under teacher forcing, as mean log-probability per token. The *copy-free margin* is the theatrical-minus-plain difference averaged over four registers (heroic archaic, sports announcer, stage melodrama, carnival barker). These theatrical answers convey the plain answer's technical content, but an automated audit confirms they share no content word or two-word phrase with any history text across all genres and 4,096 seed variants, apart from topic words that also occur in the plain answer. *Copyable* targets reuse the genre's signature phrases and serve only to measure copying.

**Estimands.** Seed-level paired contrasts, averaged over probes:
- R = ordered − neutral;
- L = lexical_match − lexical_neutral;
- F = lexical_neutral − neutral;
- **P = R − L**, the extra effect of enacting rather than quoting with format held fixed.

P is a contrast decomposition, not an identified causal effect. Quoting changes provenance, quotation and task together. Within-pair prompt-length differences are equal by construction, so P cancels length; token positions still differ.

**Mechanism.** On training seeds and training probe families, we learn the rank-one residual direction ordered − lexical_match at the final prompt token of a decoder block. On held-out seeds and probe families, we erase that direction from `ordered` prompts (projection erasure about the `lexical_match` centroid) and compare the change in copy-free margin with 8 norm-matched random orthogonal directions.

**Behaviour.** Sampled answers (T = 1, 20 per case, 64 tokens) are scored for genre intrusion and theatrical markers.

**Models.** Primary: Qwen3-4B-Instruct-2507 in bf16. Replication: Qwen3-1.7B in float32 with thinking disabled. Revisions and file hashes are pinned in the preregistration.

**Statistics.** Exact or Monte-Carlo two-sided sign-flip tests on 32 seed means, bootstrap 95% CIs, and Holm adjustment across the four confirmatory hypotheses.

## 3 Results

<!-- RESULTS:body -->

## 4 Related negative results

- **Order hypothesis.** The original question was whether persona *order* matters: does an `ordered` history differ from the same turns with the last one matched (`tail_matched`)? It failed at calibration. The contrast was small and in the wrong direction (−0.06 nat/token; Qwen3-4B, 16 seeds).
- **Copying artefact.** The first "residue" was copying of shared phrases.
- **0.6B readout confound.** An earlier 0.6B pilot scored first tokens where Qwen3's hybrid template expects `<think>`.

These are reported in `reports/`.

## 5 Limitations

- Histories are scripted, and assistant turns are never generated by the model under test.
- The readout is a forced-continuation likelihood; Q5 checks sampled behaviour only through lexical markers.
- Two small Qwen models; three genres; nine technical probe families.
- bf16 scoring of the primary model carries per-case numerical error (mean 0.05 nat/token on a float32-checked 1.7B model). It averages out at the seed level, and the float32 replication guards against it.
- P bundles provenance, quotation and task.
- Nothing here measures geometric holonomy or demonstrates a persona "inside" the model. A rank-one direction is a summary statistic, not a mechanism.

## Reproducibility

Code, case generators, targets, audits and every report are in this repository (MIT). The confirmatory run is `scripts/confirmatory_run.sh`. Model weights are not redistributed.
