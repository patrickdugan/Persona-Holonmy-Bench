# Experiment design and validity

## Research question

When the same neutral question follows different transcript trajectories, do we observe reproducible differences in response style? Does a return-to-neutral instruction reverse a prior fictional persona, and does the order of exposure matter?

## Estimands

For model \(m\), seed \(s\), neutral probe \(p\), condition \(c\) and checkpoint \(t\), let \(Y_{m,s,p,c,t}\) be a pre-registered style measure on a generated answer. The primary paired contrast is

\[
D_{m,t} = \mathbb{E}_{s,p}[Y_{m,s,p,\mathrm{ordered},t} - Y_{m,s,p,\mathrm{permuted},t}].
\]

The `ordered`/`permuted` comparison is strongest after `peak` or `exit`, when the same three induction user/assistant pairs have all appeared. The recovery contrast compares style scores at `peak`, `exit` and `washout_2`; since later checkpoints contain more transcript tokens, treat that pattern as descriptive.

The `neutral` trajectory has a similar number of conversational turns, with different lexical content. Context lengths are reported as raw character counts, **not exact model-token counts**. For tighter controls, add tokenizer-aware balancing in a model-specific follow-up.

## Fixed assistant messages

All induction turns include assistant text written by the benchmark authors. This is deliberate: independent calls can compare the model after exactly the same fixed content, without introducing a confound from different autoregressive prior completions. An interactive rollout experiment will answer a different question.

## Data integrity

- Freeze model checkpoint, tokenizer, chat template, inference engine version, decoding parameters, system prompt and evaluation scripts.
- Pair conditions by probe and seed. A model/server may ignore `seed`; record that caveat.
- Randomize actual execution order for proper runs to guard against warm-up, batching, time, and drift. The v0.1 runner uses a deterministic enumeration order; shuffle the exported cases externally or extend the runner for that design.
- Add many independent probe questions before interpreting a genre intrusion rate. Avoid training or tuning prompts against the holdout set.
- Use blinded human judgments (0=no genre intrusion, 1=minor flourishes, 2=noticeable roleplay, 3=primarily theatrical answer), plus separate ratings of technical accuracy.
- Bootstrap by independent seed (cluster), not by all rows as if each probe and seed were independent. At least five seeds are required here for displaying exploratory intervals; larger samples are needed for reliable inference.
- Keep confidence intervals descriptive, report all conditions and failures, and inspect raw answers.

## Alternative explanations

- Pure prompt echoing due to recent content or position-dependent attention
- Position and recency effects from transcript order
- Context-length effects and truncation
- Model chat template and role delimiters
- The scripted assistant turns priming the model more strongly than user turns
- System prompt compliance or contaminated baseline
- Lexical false positives (e.g. a legitimate answer referring to a cathedral)
- Server model swaps, quantization, temperature or non-deterministic kernels

We cannot infer model internals, subjective experience, or model-family identity from this protocol.

## Design upgrades

1. Increase held-out probes and add multilingual, domain-disjoint technical topics.
2. Exact model-token-match each comparison, plus equalize distances from induction turns to probe.
3. Fully counterbalance all six orders of the three induction blocks, not only one shuffled order per seed.
4. Add minimally edited content controls with the same keywords in neutral prose.
5. Randomize generation order and collect model identity/version fingerprints.
6. Compare scripted transcript injection with true interactive conversation trajectories.
7. Apply randomized, blinded ratings and preregister thresholds.

## Behavioral closure and recovery (v0.2)

With fixed neutral probe \(p\) and seed \(s\), the observable *behavioral* closure estimate is

\[
C_{c,s,p,t} = Y_{c,s,p,t} - Y_{c,s,p,\mathrm{baseline}},
\]

and the matched contrast subtracts the corresponding control trajectory:

\[
D_{s,p,t} = C_{\mathrm{ordered},s,p,t} - C_{\mathrm{permuted},s,p,t}.
\]

The analysis aggregates paired probe differences within seeds, then bootstraps seeds. This partially addresses baseline imbalance but does not remove differences in the positional distribution of prior text. In particular, these are **linguistic response metrics** and are not estimates of a connection, parallel transport, or holonomy on a learned state space.

Generation is now randomized using a recorded independent execution seed to reduce correlation with backend warm-up and chronological drift. Models may ignore per-case sampling seeds. Do not claim empirical invariance without repeats using a pinned model, tokenizer and inference runtime.

## Last-turn-matched control (v0.3)

For a three-turn induction [A, B, C], the new `tail_matched` condition [B, A, C] preserves the multiset of turns and the final induction turn's position. This guards against attributing all order effects to the most recent Gothic cue. Longer histories will require more than one matched permutation to test generality. Token identities and positions in the earlier turns still differ, so effects may be positional rather than a stable persona attractor.

## Open-weight activation measurement

An optional hidden-state forward-pass adapter exports selected layer outputs at the final input token and records exact token counts. The first measurement is a pairwise cosine distance with baseline and `tail_matched`, with matched seed and neutral probe. See [activation protocol](activation-protocol.md); these are descriptive statistics and make no claim about geometric parallel transport.
