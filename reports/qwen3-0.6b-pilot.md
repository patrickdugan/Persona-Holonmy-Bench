# Qwen3-0.6B v0.13 workstation pilot (2026-10-23)

**Shakedown only.** This is the first PHB pipeline run on pretrained weights. Three held-out seeds cannot give a two-sided exact sign-flip p-value below 0.25. Nothing here supports a claim about persona transfer, percolation advantage, or holonomy.

Summary: `Qwen checkpoint/revision: Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca; stages completed: capture, fit, graph, select, map, evaluate (pilot-002); CI: pending, not pushed from the workstation; science gates: block-14 SAE fails reconstruction (validation FVU 1.01), all seven selectors coincide, later-token effect structurally zero; evidence: the training-selected block-27 latents lower the held-out style margin more than norm-matched orthogonal nulls in 3/3 seeds (+0.052 log-odds, p = 0.25, the floor); next action: larger SAE corpus and candidate pool, plus a non-final block, before any selector comparison.`

## Checkpoint and environment

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` (`Qwen3ForCausalLM`, 28 blocks, hidden 1024), local HF cache snapshot, no download |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` (snapshot directory = `refs/main`) |
| Weights SHA-256 | `model.safetensors` `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`, equal to the upstream LFS oid at that revision (HF tree API) |
| Tokenizer | `tokenizer.json` SHA-256 `aeb13307…2dae4` matches upstream LFS oid; `config.json`, `tokenizer_config.json`, `generation_config.json`, `vocab.json`, `merges.txt` git-blob SHA-1s match upstream |
| Code | branch `qwen3-0.6b-pilot` from `9cd077e` (v0.13), plus the `graph_hf` fix below |
| PercInterp | `c0b3f6abf8e847292aaf1e21b8f4b804fd4d8ce0` (CI pin) |
| Python / libs | CPython 3.12.10, torch 2.11.0+cu128, transformers 5.17.0, numpy 2.5.3, scipy 1.18.1, safetensors 0.8.0, pytest 9.1.1 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, driver 577.05; GPU stages run under a 4 GB `vramorama` lease |
| Peak VRAM | 1514 MiB (graph), 1748 MiB (evaluate); whole-card `nvidia-smi` samples at 500 ms, nothing else running on the card |
| dtype / device | float16 CUDA for capture/graph/evaluate; SAE fit on CPU (runner defaults) |
| Tests | `python -m pytest`: 86 passed, 0 skipped (85 at v0.13 plus one regression test) |

Environment note: the C: drive had about 270 MB free, so the venv borrows torch/transformers from a sibling project's venv through a `.pth` file. It does not run `pip install -e '.[test,mechanistic]'`. PercInterp (pinned commit), scipy and PHB (`-e .`) are installed in the venv itself.

## Commands

```bash
phb pilot-plan --model '<HF cache>\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca' \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --output-dir 'runs\qwen-pilot-002' \
  --repetitions 8 --layers mid,last --top-features 4 --random-controls 4 \
  --max-total-tokens 1536 --max-tokens-per-case 32 --max-model-forwards 250
# GPU stages (capture, graph, evaluate) were wrapped as:
vramorama run --vram 4G --owner phb-pilot-<stage> -- .venv/Scripts/python.exe -u -m persona_holonomy pilot-stage --plan 'runs\qwen-pilot-002\plan.json' --stage <stage>
# CPU stages (fit, select, map) ran without the wrapper.
```

Stages ran one at a time, and each stage's outputs were checked before the next. Wall time per stage, including about 17 s of lease admission: capture 36 s, fit 15 s, graph 32 s, select and map a few seconds each, evaluate 38 s.

## Run history: pilot-001 was stopped, pilot-002 is the result

`runs/qwen-pilot-001` completed capture and fit. Its graph stage then crashed before any measured forward: `TypeError: graph_plan() got an unexpected keyword argument 'max_total_tokens'` (traceback in `qwen3-0.6b-pilot-002/logs/pilot-001-graph-TypeError.log`). `graph_hf` passed every CLI keyword, including the measurement-only `--max-total-tokens`, into the offline `graph_plan` preflight. The fix filters that keyword out, the same way `component_ablation_hf` already filters its preflight kwargs. A regression test drives `graph_hf` with the CLI's token budget. It fails on v0.13 and passes now. The CPU-toy CI never reached this path, because the toy tests call `measure_graph` directly.

Following the runner's guard, pilot-001 was kept unmodified (with its partial `attribution-plan.json`) and the pilot restarted in `runs/qwen-pilot-002` with identical parameters. Reproduction was exact: `cases.jsonl` (`19c399fb…bcbe6`), both activation archives and both SAE checkpoints have identical SHA-256s in 001 and 002. The frozen split and test partition were not touched.

## Split (frozen, verified at each stage)

| | Seeds | Probe family |
|---|---|---|
| PHB train | 100, 101, 102, 104, 105 | `flaky_ci` |
| SAE fit / validation (subset of train) | 100, 101, 104, 105 / 102 | `flaky_ci` |
| Held-out evaluation | 103, 106, 107 | `parser_validator` |

There are 32 cases (8 seeds × ordered/tail_matched × 2 probes × `exit`). Capture, registry, graph provenance, selection provenance and every ablation row all carry the same train and test seed and probe lists. No test seed or probe appears in any training artifact.

## Stage results

**Capture.** Blocks 14 (`mid`) and 27 (`last`), training cases only: 256 fit tokens and 64 validation tokens per block (32 per case), all finite in fp16. Max |activation| is 38 at block 14 and 316 at block 27.

**Fit (gate: weak).** TopK SAEs with k=8 and 4,096 latents, trained for 100 steps on 256 tokens (the registry warns that the corpus is smaller than the dictionary).

| Block | fit FVU | validation FVU | validation coverage |
|---|---|---|---|
| 14 | 0.826 | **1.014** | 0.028 |
| 27 | 0.519 | 0.604 | 0.021 |

The block-14 SAE does not reconstruct held-out tokens better than their mean, so its latents have no validated meaning. The block-27 SAE is weak. This is the main limit on everything downstream.

**Graph.** There are 4 candidate latents per block, chosen by training-only ordered-vs-tail contrast. The run used 50 forwards (the plan allowed at most 50 and the stage cap was 250) and emitted 20 signed edges: every 14→27 pair and every 27→output edge, so the graph is complete bipartite. `graph_status = spanning`, and there are no skipped zero-contrast patches. Edge magnitudes are small: 14→27 edges are |w| ≤ 0.007 in SAE-activation units, and 27→first-token style log-odds edges are −0.0055, +0.0141, +0.0055 and −0.0055. Because the graph contains every possible edge, "spanning" says nothing about structure.

**Select (gate: degenerate).** The sieve threshold is c\* = 1.03, the shuffle p-value 0.21 (32 shuffles) and z_peak 1.03, below z_min 3. The stage quota is 4 latents at block 27, and the pool has exactly 4 candidates there, so **sieve, local_degree, top_attribution and random_000..003 all select the same set** `sae:27:{1944,2200,2308,3020}` (`selector_identical_to_sieve` is true for all of them). The percolation-vs-comparator question is therefore untestable in this configuration. This is a design limitation of `--top-features 4` with a two-block graph, not evidence either way.

**Map.** Four nodes map to the pinned block-27 SAE (`48931b96…74e`).

**Evaluate.** There are 3 held-out cells (ordered history, `parser_validator`, seeds 103/106/107) and 99 forwards out of a cap of 250. The planner found 1 unique component set across 7 selectors. Controls held on every cell:
- Alpha-zero sham patch equals baseline within 1e-5 (the runtime self-check raises otherwise).
- Each orthogonal null's norm equals its candidate's within 1e-4 (both about 455.8).
- Identical held-out cells for every selector.

Readout: margin is logit(theatrical) − logit(plain) over the fixed continuations. *specificity = orthogonal margin − candidate margin*, so a positive value means ablating the selected latents lowers the theatrical preference more than a same-norm random direction does.

| Seed | baseline margin | candidate | orthogonal null | candidate effect | null effect | specificity | first-token: base / cand / null | later-token (all three) |
|---|---|---|---|---|---|---|---|---|
| 103 | −0.544 | −0.705 | −0.644 | 0.161 | 0.100 | +0.061 | 4.54 / 3.27 / 2.87 | −1.181 |
| 106 | −0.449 | −0.611 | −0.531 | 0.162 | 0.082 | +0.080 | 4.46 / 3.15 / 4.40 | −1.083 |
| 107 | −0.527 | −0.685 | −0.671 | 0.159 | 0.145 | +0.014 | 4.62 / 3.39 / 4.06 | −1.165 |

- Mean specificity is **+0.052** log-odds, positive in 3/3 seeds. The within-selector exact sign-flip test gives p = 0.25, the minimum possible with 3 clusters. Paired selector-vs-sieve tests give p = 1.0 and Holm 1.0 (all selectors identical).
- The candidate effect is very stable (0.159–0.162). The null effect varies 0.08–0.15, and the first-token null ranges from 2.87 to 4.40. With a single null draw per cell, most of the specificity variance comes from the null.
- First-token specificity is mean +0.51 with mixed sign (−0.40, +1.25, +0.67).
- **Later-token specificity is exactly 0, by construction:** the patch hits block 27, the final decoder block, at the final prefix position. No later layer reads that position's block-27 residual, so only the first predicted token can change. The first-token/later-token split is uninformative for `last`-block patches. A later-token test needs a selected block below the top.

## Reading

The four training-selected block-27 SAE latents carry a small, seed-consistent share of the theatrical-vs-plain first-token preference on an unseen probe family. That share is slightly larger than what a norm-matched random direction removes. The effect is small, rests on 3 seeds at the p-value floor, uses an SAE with 0.60 validation FVU fitted on 256 scripted tokens, and comes from a forced-continuation proxy. It does not test whether percolation selection beats degree or random selection, because the comparators were identical. No free-generation or blinded check was run.

## Recommended next pilot (not run)

1. Grow the SAE corpus (more seeds, more tokens per case, more phases) or reduce `n_latents` well below the token count. Re-gate on validation FVU per block before graph estimation.
2. Use `--top-features` well above the per-block stage quota so selectors can differ. Confirm `n_unique_component_sets > 1` in `component-plan.json` before spending GPU on evaluation.
3. Put the selected block below the final block (for example `mid` plus a block in between) so later-token effects are measurable.
4. Average several orthogonal null draws per cell; a single draw dominates the variance here.
5. Preregister a larger held-out seed set (about 30, per HANDOFF) before treating any sign-flip p-value as evidence.

## Artifacts

`reports/qwen3-0.6b-pilot-002/` holds byte-identical copies of the pilot-002 JSON outputs (their SHA-256s match `progress.json`) and the stage logs. Activation archives (`*.npz`) and SAE checkpoints (`*.pt`) stay local under `runs/qwen-pilot-002/` (66 MB) per the data boundary. They can be regenerated deterministically with the commands above.
