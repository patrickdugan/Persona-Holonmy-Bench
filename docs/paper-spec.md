# Paper spec: the persona likelihood residue (option A, 2026-10-24)

**Status:** design, not preregistered. Calibration numbers come from `reports/h1-calibration.md` and `reports/lexical-match-calibration.md`. Those seed ranges (9000–9015 and 9100–9115) are spent and must not be reused for confirmation.

**Working title:** *Persona residue: likelihood carryover without behavioural carryover after a role-play exit in small chat LLMs.*

## Phenomenon

Qwen3-4B-Instruct-2507 and Qwen3-0.6B are given a scripted three-turn persona history (gothic fantasy game), then an explicit "we have finished the previous topic" turn. Both then answer unrelated technical questions plainly: 0 of 192 greedy answers per model contain any persona marker. Yet a theatrical continuation of the answer stays about **+1 nat per token** more likely than after a persona-free `neutral` history. This held in 16/16 seeds, in both models, and at both `exit` and `washout_2`.

The original hypothesis, that persona *order* matters (`ordered` > `tail_matched`), failed: that contrast is small and negative. The paper therefore studies the residue itself: what produces it, whether it lives in identifiable features, and whether it is behaviourally live but suppressed.

## Questions and estimands

Every estimand is a seed-clustered paired contrast in the forced-continuation margin (theatrical mean log-prob per token minus plain), averaged over probes within each seed. First-token and later-token parts are reported separately.

- **Q1 – Residue.** R = margin(`ordered`) − margin(`neutral`) at `exit`, on fresh seeds and on probe families never used in calibration. In the same cells, greedy free generation must stay plain: genre intrusion ≤ 2% in every condition. That condition is what makes this "likelihood without behaviour".
- **Q2 – Decomposition.** Two opt-in controls, both now in `protocol.py`:
  - `lexical_match` quotes every induction request and reply verbatim, in order, inside user turns for a neutral sentence-counting task. They appear under neutral labels `A:`/`B:`, and the assistant only answers "A: n. B: n.".
  - `lexical_neutral` wraps the neutral turns in the same counting format.

  The estimands:
  - **L** = `lexical_match` − `lexical_neutral`: the effect of the theatrical tokens being in context, with format held fixed.
  - **P** = (`ordered` − `neutral`) − (`lexical_match` − `lexical_neutral`): a difference-in-differences. It is the extra residue when the assistant *wrote in persona* under a role-play request, rather than the same text being quoted as data.

  This is a contrast decomposition, not an identified causal one. Moving text into a quoted counting task bundles three changes: who wrote it (provenance), whether it was quoted, and the task. Token positions also differ: the wrapper lengthens each turn and moves the theatrical text relative to the probe. Prompt length is reported for every cell and used as a seed-level covariate in a sensitivity analysis.
- **Readout overlap.** The original theatrical targets reuse "ACT IV" and "QUEST UNLOCKED" from the gothic reply, so their later-token margin can be phrase copying. Every estimand is therefore also computed on **novel theatrical targets** that share no phrase or setting word with any induction text (`NOVEL_THEATRICAL`, currently calibration-only in `scripts/h1_calibration.py`; it moves into the package with preregistered wording). The confirmatory readout is the novel-target margin. The original targets are reported for continuity.
- **Q3 – Persistence and suppression.**
  - Persistence: compare R across `exit`, `washout_1` and `washout_2`.
  - Suppression: compare R at `peak` (no exit turn) with R at `exit`. The ratio is how much of the residue the exit instruction suppresses.
  - Behaviour: greedy genre intrusion at `peak` vs `exit` shows whether the residue is behaviourally live before the exit turn.
- **Q4 – Mechanism.** On training seeds only, learn the rank-one direction (`ordered` − `neutral`, and separately `ordered` − `lexical_match`) at preregistered blocks. On held-out seeds:
  - **(a)** projection erasure must shrink R (respectively P) more than 16 norm-matched orthogonal nulls do;
  - **(b)** injecting the direction into a `neutral` history must raise the margin dose-dependently, and greedy outputs are checked for persona intrusion. This tests whether the direction carries the residue and whether it is behaviourally potent once the exit suppression is bypassed.
- **Q5 – Features (secondary, gated).** TopK SAE (or real VPD, see below) features at the Q4 blocks. Do percolation-selected features account for more of R or P than equal-budget degree, attribution and random selections? This runs only if every gate below passes; otherwise it is reported as not tested.

**Confirmatory family:** Q1, Q2-P, Q2-L, Q3-suppression and Q4a, all on the novel-target margin. Holm-adjusted at α = 0.05. Q4b and Q5 are secondary; Q3-persistence is descriptive.

## Design

| Item | Plan |
|---|---|
| Primary model | Qwen3-4B-Instruct-2507 @ `cdbee75` (non-thinking), bf16, local 16 GB card |
| Replication | Qwen3-0.6B and Qwen3-1.7B with the chat template's `enable_thinking=False` (needs a template-kwargs option in PHB; the current code renders the default thinking template). A second model family would need a download, so ask first. |
| Personas | The current gothic-game genre plus **two new genres** (for example pirate radio drama, noir detective), each with its own theatrical continuations, so the residue isn't one genre's lexicon |
| Probe families | Grow from 3 to **9 technical families**: 3 for training (direction or feature learning) and 6 held out, each with a plain/theatrical continuation pair per genre |
| Seeds | Confirmatory range 20000–20063: 32 training seeds, 32 held out. SAE validation uses a subset of the training seeds. |
| Conditions | `ordered`, `neutral`, `lexical_match`, `lexical_neutral` (primary); `tail_matched` and `permuted` (secondary, for continuity with v0.13) |
| Phases | `peak`, `exit`, `washout_1`, `washout_2` |
| Statistics | Exact or Monte-Carlo sign-flip tests on seed means; bootstrap 95% CIs; Holm over the confirmatory family; raw and adjusted p reported |
| Power | Calibration seed SDs (R about 0.07; P and L from the lexical-match calibration) give effects far above detection at 32 held-out seeds. The binding constraint is the smallest effect of interest: P or L counts as present at ≥ 0.05 nat per token. |

## Gates (each recorded as pass or fail; a failure is reported, never bypassed)

1. A placebo at `baseline` gives exactly 0 for every contrast.
2. No train/held-out leakage in seeds, probe families or genres, checked by fingerprints.
3. The alpha-0 identity and the null-norm match hold on every intervention cell.
4. Q5 only: SAE validation FVU ≤ 0.3 at each block, on both PHB text and generic text; selector sets differ (Jaccard < 0.5 against the sieve); sieve shuffle p < 0.05 on the training graph.
5. Model-forward budgets are fixed in `plan.json` before running.

## Work order

1. **Code** (delegable):
   - template kwargs (`enable_thinking=False`);
   - the two new persona genres;
   - 6 new probe families with continuations;
   - `lexical_match` for the new genres;
   - multi-null averaging;
   - a residue-decomposition report command.
   Tests for each.
2. **Calibration** on a fresh calibration range (9200+) to fix blocks and confirm the gates are reachable. Not reported as evidence.
3. **Preregistration:** `docs/preregistration.md` holding hypotheses, estimands, gates, seeds, split, analysis and Holm family, plus the frozen plan hashes. Commit it before any held-out run, and cite the commit in the paper.
4. **Confirmatory runs:** Q1–Q4 on the primary model, then the replications.
5. **Q5** if its gates pass. VPD on a 2-layer slice of the 4B model would need an 80 GB pod, about $75–150 for two seeds (see the cost note in this repo's history). Ask before renting.
6. **Write-up:** 4–6 pages in `paper/`.

## Paper outline

1. **Intro:** role-play exits look clean behaviourally, yet a measurable likelihood residue remains. Safety relevance: suppressed but recoverable persona state.
2. **Setup:** histories, controls (especially `lexical_match`), readouts, models.
3. **Results:**
   - Q1: residue without behaviour.
   - Q2: decomposition into persona framing and lexical priming.
   - Q3: suppression by the exit turn and persistence.
   - Q4: a direction carries it and is behaviourally potent when injected.
   - Q5: features and selectors, or "not tested" with the failed gate.
4. **Negative results:** the failed order hypothesis (`ordered` < `tail_matched`); the 0.6B thinking-template confound.
5. **Limitations:**
   - scripted histories;
   - a forced-continuation proxy;
   - small models;
   - no claim of geometric holonomy;
   - a rank-one direction is not a persona.

## Done when

- The preregistration commit precedes every held-out run.
- Every gate is reported.
- Q1–Q4 each have estimates, CIs and Holm-adjusted p on the primary model plus at least one replication.
- CI is green on the cited commit.
- The draft is in `paper/`, with run records in `reports/`.
