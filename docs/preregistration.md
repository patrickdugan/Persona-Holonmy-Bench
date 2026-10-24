# Preregistration: enacted vs quoted persona text (PHB, 2026-10-25)

This file is committed **before** any confirmatory model run. The commit hash that introduces it is the preregistration timestamp cited in the paper. Changes after that commit are listed under "Deviations", with reasons. The plan behind it is `docs/paper-spec.md`; the calibration evidence is in `reports/h1-calibration.md`, `reports/lexical-match-calibration.md` and `reports/cal9200.md`.

## Models

| Role | Checkpoint | Settings |
|---|---|---|
| Primary | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` | bf16, CUDA, default chat template |
| Replication | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | **float32**, CUDA, `--no-thinking` (`enable_thinking=False`) |

All weight and tokenizer files were verified against the upstream LFS SHA-256 / git blob hashes at these revisions.

**Numerics:** on Qwen3-1.7B, bf16 scoring differs from float32 by a mean of 0.05 and at most 0.22 nat per token per case margin. This holds for both the cached and the uncached scorer (`runs/fastcheck/check_ref.py`). The primary model must run in bf16 to fit the 16 GB card. The replication therefore runs in float32 as an independent full-precision check.

**Commands:** every confirmatory command is in `scripts/confirmatory_run.sh`, committed with this file.

## Materials (frozen at this commit)

- **Protocol:** `protocol.py` with `variation=seeded_v1`.
  - **Genres:** gothic, pirate, noir.
  - **System variants:** default, plain.
  - **Conditions:** ordered, neutral, lexical_match, lexical_neutral.
- **Confirmatory probe families:** the six extra families `retry_backoff`, `schema_migration`, `memory_leak`, `rate_limit`, `feature_flag`, `log_rotation`. No calibration ever used them. The three original families (`flaky_ci`, `parser_validator`, `stale_cache`) are training-only (Q6 direction learning).
- **Readouts:** `CausalProbe.all_margins` (prefix-cached batched scorer).
  - **Primary:** `copy_free_margin`, the mean over the four registers in `targets.COPY_FREE_TARGETS` of the theatrical-minus-plain mean log-prob per token.
  - **Secondary:** `margin` (gothic-style copyable targets) and `copyable_margin` (genre-specific copyable targets).
- **Seeds:** 20000–20031 (32 seeds). Calibration seed ranges 9000–9015, 9100–9115 and 9200–9215 are excluded.

## Estimands

A seed-level value is the contrast averaged over the six confirmatory probe families and, where stated, over the three genres. Definitions are in `decomposition.py`:

- R = ordered − neutral
- L = lexical_match − lexical_neutral
- F = lexical_neutral − neutral
- P = R − L

## Confirmatory hypotheses

All are two-sided exact or Monte-Carlo sign-flip tests on 32 seed means (`inference.signflip_pvalue`), Holm-adjusted across these four:

| ID | Hypothesis | Cell | Prediction |
|---|---|---|---|
| H1 | Provenance effect P ≠ 0 | 4B, `exit`, default system, pooled over genres | P < 0 |
| H2 | Quoted priming L ≠ 0 | same | L > 0 |
| H3 | System-prompt effect on P: ΔP = P(plain) − P(default) ≠ 0 | 4B, `peak`, pooled over genres | none preregistered; the sign decides whether suppression is instruction-driven (ΔP > 0) |
| H4 | Direction specificity ≠ 0 | 4B, gothic, default system, `exit`, block 18 | > 0 (raw: erasure raises the margin more than nulls) |

**H4 detail:** H4 uses its own case file, seeds 20000–20063 × all nine probe families. The sweep's deterministic split is `--holdout-fraction 0.5 --split-seed 20261025 --heldout-probes` (the six confirmatory families). The rank-one direction `ordered − lexical_match` is learned only on training seeds × the three original families, with `phb sweep --sources ordered --reference lexical_match --readout copy_free --n-nulls 8 --alphas 1 --layers 18`. Specificity is defined on **raw** effects (patched minus baseline copy-free margin) on held-out ordered cases (held-out seeds × confirmatory families): the erasure effect minus the mean of the 8 norm-matched orthogonal-null effects, as a seed-level mean. H4 predicts specificity > 0: erasing the enacted-vs-quoted direction raises theatrical likelihood. In the sweep's own sign convention (`oriented_effect`, where positive means less theatrical under erasure) this shows up as a negative `mean_paired_specificity`. The number of held-out seeds is fixed by the split and reported.

A hypothesis is **confirmed** when its Holm-adjusted p < 0.05 and the effect has the predicted sign (H3: either sign is reported). Effects with |mean| < 0.10 nat per token are called "small" whatever their p.

## Secondary and descriptive analyses (no multiplicity claim)

- **Per genre:** H1/H2 within each genre; the copying share (Q3) per genre with `copyable_margin`.
- **Time course:** P at `peak`, `exit`, `washout_1` and `washout_2` (default system).
- **Q5 behaviour:** `phb sample-rates`, n = 20 samples at T = 1.0, 64 new tokens, at `exit`, default system, seeds 20000–20015, all genres and the four conditions. Reports genre-intrusion and theatrical rates by condition.
- **Q6 extra blocks:** blocks 9 and 27 (same sweep) and injection into held-out `lexical_match` cases.
- **Replication (1.7B):** H1–H3 cells only.
- **Q7 (SAE/VPD features):** not run in this round. The SAE gate requires a generic-text corpus and VPD requires a paid pod; both are reported as not tested.

## Gates (reported pass/fail; a failure is reported, not bypassed)

1. A `baseline`-phase placebo on seeds 20000–20003: every contrast is exactly 0.
2. `targets.audit_copy_free()` returns no violations.
3. Within-pair prompt-length differences are equal for ordered/neutral and lexical_match/lexical_neutral (DiD cancels length).
4. The sweep's alpha-0 identity and null-norm checks pass on every cell.
5. No case exceeds `max_total_tokens` = 2048. A case that would is reported, not truncated.

## Exclusions and stopping

No exclusions. Every scheduled cell is run once. A crashed run is resumed by restarting the affected (genre, model) job from scratch in a new directory; partial outputs are kept and reported.

## Deviations

None yet.
