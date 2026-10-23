# Paper spec: enacted vs quoted persona text (option A, revised 2026-10-24)

**Status:** design, not preregistered. Calibration numbers come from `reports/h1-calibration.md` and `reports/lexical-match-calibration.md`. Seed ranges 9000–9015 and 9100–9115 are spent and must not be reused for confirmation.

**Working title:** *Enacted personas are sealed off, quoted ones leak: provenance-dependent style priming after role-play in small chat LLMs.*

## What the calibrations showed

The spec was first written around a "likelihood residue": after a role-play history, theatrical continuations stayed about 1 nat per token more likely than after a neutral history, with no change in behaviour. The `lexical_match` / `lexical_neutral` controls and copy-free targets dissolved that framing (Qwen3-4B-Instruct, 16 seeds, 3 probes):

1. **The residue was phrase copying.** The original theatrical targets reuse "ACT IV" and "QUEST UNLOCKED" from the history. Their residue is the same whether the text was enacted by the assistant or quoted as data (difference-in-differences −0.08 at `exit`). On theatrical targets with no shared phrase, it reverses sign (ordered − neutral = −0.44).
2. **Provenance dissociation:**
   - Theatrical text quoted inside a user turn primes a copy-free theatrical style: L = +0.33 nat per token, format held fixed.
   - The same text enacted by the assistant in a role-play *suppresses* that style: P = −0.77 relative to the quoted version, in 16/16 seeds and all three probes.
   - The effect is largest straight after the fiction (−0.92 at `peak`), already present without the exit turn, and fades over two neutral turns (−0.41).
3. **No behavioural difference at greedy decoding:** 0 of 384 answers show persona markers in any condition.

The paper therefore asks how *who produced* persona text (the assistant in role, or a user as quoted data) changes later style likelihood, and what carries that difference inside the model.

## Questions and estimands

Every estimand is a seed-clustered paired contrast in the **copy-free** margin: the mean log-prob per token of a theatrical continuation sharing no phrase or setting word with the history, minus the plain continuation. It is averaged over a preregistered set of copy-free targets and over probes within each seed. First-token and later-token parts are reported separately. The original copyable targets are reported only to quantify copying.

- **Q1 – Provenance effect (primary).** P = (`ordered` − `neutral`) − (`lexical_match` − `lexical_neutral`) at `exit`. Prediction: P < 0.
- **Q2 – Quoted priming.** L = `lexical_match` − `lexical_neutral`. Prediction: L > 0.
- **Q3 – Copying share.** The original-target margin minus the copy-free margin, per condition. P on the original targets is expected to be about 0: copying does not depend on provenance.
- **Q4 – What drives the suppression.**
  - Time course: P at `peak`, `exit`, `washout_1` and `washout_2`.
  - Two new manipulations:
    - **(a) `system_plain`:** a system prompt without the clause "When a fictional exchange ends, respond to the next request normally".
    - **(b) `no_exit`:** `peak` already omits the exit turn.

    If P at `peak` with `system_plain` goes to about 0 or turns positive, the suppression is instruction-driven. If it stays negative, the model closes off enacted style without being told to.
- **Q5 – Behaviour.** Sampled decoding (T = 1, 20 samples per case) at `peak` and `exit`. Does the rate of theatrical answers follow P and L? This is what links the likelihood shifts to behaviour. Greedy decoding showed nothing.
- **Q6 – Mechanism.** On training seeds, learn the rank-one direction for the enacted-vs-quoted contrast (`ordered` − `lexical_match`, centred on the format-matched neutral pair) at preregistered blocks. On held-out seeds:
  - **(a)** projection erasure must move P toward 0 more than 16 norm-matched orthogonal nulls do;
  - **(b)** injecting the direction into `lexical_match` histories must lower the copy-free margin, dose-dependently.
- **Q7 – Features (secondary, gated).** TopK SAE features, or real Goodfire VPD components, at the Q6 blocks. Do percolation-selected features account for more of P than equal-budget degree, attribution and random selections? This runs only if every gate below passes; otherwise it is reported as not tested.

**Confirmatory family:** Q1, Q2, Q4(a) and Q6(a), Holm-adjusted at α = 0.05. Q3 and the Q4 time course are descriptive; Q5, Q6(b) and Q7 are secondary.

## Design

| Item | Plan |
|---|---|
| Primary model | Qwen3-4B-Instruct-2507 @ `cdbee75` (non-thinking), bf16, local 16 GB card (about 8 GB peak) |
| Replication | Qwen3-0.6B and Qwen3-1.7B with the chat template's `enable_thinking=False` (needs a template-kwargs option in PHB; the current code renders the default thinking template, which confounded the 0.6B readouts). A second model family would need a download, so ask first. |
| Personas | The current gothic-game genre plus **two new genres** (for example pirate radio drama, noir detective) |
| Copy-free targets | Per probe and genre, **4 theatrical continuations in different registers** (archaic, announcer, melodrama, noir). Automated check: no word of 4+ letters and no 2-word phrase shared with any history text. Written and frozen before preregistration. |
| Probe families | Grow from 3 to **9**: 3 for training (direction or feature learning), 6 held out |
| Conditions | `ordered`, `neutral`, `lexical_match`, `lexical_neutral` (primary); `tail_matched`, `permuted` (continuity); `system_plain` variants of all four primaries (Q4a) |
| Phases | `peak`, `exit`, `washout_1`, `washout_2` |
| Seeds | Confirmatory range 20000–20063: 32 training, 32 held out. Calibration 9200+. |
| Statistics | Exact or Monte-Carlo sign-flip on seed means; bootstrap 95% CIs; Holm over the confirmatory family; raw and adjusted p reported |
| Power | Calibration seed SDs are 0.03–0.06 for P and L; the per-probe spread (−0.29 to −1.64) is the larger variance. Smallest effect of interest: 0.1 nat per token. With 32 held-out seeds and 6 held-out probe families, power is high for seed-level effects. Probe-family generalisation is checked by a held-out-family random-effects estimate. |

## Gates (each recorded as pass or fail; a failure is reported, never bypassed)

1. A placebo at `baseline` gives exactly 0 for every contrast.
2. The copy-free target check passes for every target × history.
3. Within each pair (`ordered`/`neutral` and `lexical_match`/`lexical_neutral`), prompt-length differences match, so the DiD cancels length. This held exactly in calibration.
4. No train/held-out leakage in seeds, probe families or genres, checked by fingerprints.
5. Alpha-0 identity and null-norm match on every intervention cell.
6. Q7 only: SAE validation FVU ≤ 0.3 at each block, on both PHB text and generic text; selector sets differ (Jaccard < 0.5 against the sieve); sieve shuffle p < 0.05 on the training graph.
7. Model-forward budgets are fixed in `plan.json` before running.

## Work order

1. **Code** (delegable to Codex with review):
   - template kwargs (`enable_thinking=False`);
   - a `system_plain` system-prompt variant;
   - two new persona genres;
   - 6 new probe families;
   - copy-free target sets with the overlap checker;
   - move `NOVEL_THEATRICAL` and the margin into the package;
   - sampled-decoding intrusion rates;
   - multi-null averaging;
   - a decomposition report command (P, L, copying share).
   Tests for each.
2. **Calibration** (9200+) of the new genres, targets and `system_plain`, and choice of the Q6 blocks. Not reported as evidence.
3. **Preregistration:** `docs/preregistration.md` plus frozen plan hashes, committed before any held-out run.
4. **Confirmatory runs:** Q1–Q6 on 4B, then the replications.
5. **Q7** if its gates pass. VPD on a 2-layer slice of 4B would need an 80 GB pod, about $75–150 for two seeds. Ask before renting.
6. **Write-up:** 4–6 pages in `paper/`.

## Paper outline

1. **Intro:** role-play exits look behaviourally clean. Measured on likelihoods, persona text the model *enacted* is suppressed afterwards, while the same text *quoted* by a user primes the style. Safety relevance: models track who produced persona text, which bears on persona injection via quoted content.
2. **Setup:** histories, the quoted controls, copy-free targets, readouts, models.
3. **Results:**
   - Q3: the apparent residue is copying.
   - Q1/Q2: the provenance dissociation.
   - Q4: what drives the suppression.
   - Q5: does it reach behaviour?
   - Q6: the direction that carries it.
   - Q7: features and selectors, or "not tested" with the failed gate.
4. **Negative results:**
   - the original order hypothesis (`ordered` < `tail_matched`);
   - the copying artefact in the first readout;
   - the 0.6B thinking-template confound.
5. **Limitations:**
   - scripted histories;
   - the DiD bundles provenance, quotation and task;
   - forced-continuation proxies;
   - small models;
   - no claim of geometric holonomy or of a persona inside the model.

## Done when

- The preregistration commit precedes every held-out run.
- Every gate is reported.
- Q1, Q2, Q4(a) and Q6(a) each have estimates, CIs and Holm-adjusted p on the primary model plus at least one replication.
- CI is green on the cited commit.
- The draft is in `paper/`, with run records in `reports/`.
