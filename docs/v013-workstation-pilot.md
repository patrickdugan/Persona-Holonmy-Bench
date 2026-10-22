# PHB v0.13: resumable, budget-capped workstation pilot

**Status:** offline planner and stage runner are implemented and tested with synthetic stage outputs. **No pretrained Qwen result is reported.** This runner is a reproducibility guard around existing PHB v0.11–v0.12 commands, not a new causal estimator.

## Goal

Run a small end-to-end Qwen experiment using the **same frozen, seeded PHB cases** for SAE acquisition, signed finite-difference graph construction, PercInterp selection, and held-out feature-contribution ablation. Preserve the training/test boundary, avoid silent model downloads, and stop if artifacts or inference budgets change unexpectedly.

The workstation is expected to have a **trusted, cached Hugging Face causal chat checkpoint** in a local directory containing `config.json`, tokenizer files and weights, plus a functional PyTorch CUDA environment and the pinned [PercInterp](https://github.com/MoralityLabAI/percolation-interp) package. A GGUF or Ollama model name is not a valid Hugging Face path. Run with Python 3.11+ for the PercInterp dependency.

## Freeze the pilot (no GPU, no weights needed)

```powershell
python -m pip install -e '.[test,mechanistic]'
# Install the pinned PercInterp commit referenced in .github/workflows/ci.yml.
python -m pytest

phb pilot-plan `
  --model 'C:\models\qwen-local-hf' `
  --revision 'PASTE_REAL_40_HEX_COMMIT_OR_64_HEX_WEIGHT_DIGEST' `
  --output-dir 'runs\qwen-pilot-001' `
  --repetitions 8 --layers mid,last `
  --top-features 4 --random-controls 4 `
  --max-total-tokens 1536 --max-tokens-per-case 32 `
  --max-model-forwards 250
```

The model/revision values must identify **the same weights** used by capture, graph estimation and ablation. The runner requires a revision declaration in hex. Record how it was verified against the local snapshot; it does not cryptographically authenticate Hugging Face weights.

`plan.json`, `cases.jsonl` and `progress.json` are created in the new output directory. The cases contain only scripted public prompts. `plan.json` records the train/test seed partition, SAE fit/validation seeds, training/held-out probe families, input hash and a conservative graph-forward estimate. The default 8-seed plan produces five PHB training seeds (one technical question), three held-out evaluation seeds (a different technical question), and a subset of the training seeds reserved for SAE reconstruction validation. **This is an engineering shakedown, not a powered statistical study.**

## Execute one stage at a time

Run the next command only after inspecting the preceding stage's outputs. Each step is an explicit opt-in. The commands default to CUDA/float16 for the model stages, while SAE fitting uses CPU to avoid simultaneous model and SAE VRAM demands.

```powershell
$P = 'runs\qwen-pilot-001\plan.json'
phb pilot-stage --plan $P --stage capture
# Inspect capture/capture-manifest.json and token counts.
phb pilot-stage --plan $P --stage fit
# Inspect saes/sae-registry.json and validation FVU / feature coverage.
phb pilot-stage --plan $P --stage graph
# Inspect attribution-plan.json and attribution-graph.json, including edge reliability.
phb pilot-stage --plan $P --stage select
# Inspect perc-selection.json, graph significance, and comparator overlap.
phb pilot-stage --plan $P --stage map
# Inspect feature-map.json and each SHA-256-pinned SAE reference.
phb pilot-stage --plan $P --stage evaluate
# Inspect component-report.json and held-out effect sizes.
```

The stages call the existing commands, in order:

| Stage | Underlying PHB commands | Output markers |
|---|---|---|
| `capture` | `sae-capture` | `capture/capture-manifest.json` |
| `fit` | `sae-fit` | `saes/sae-registry.json` |
| `graph` | `sae-attribution-plan`, `sae-attribution` | `attribution-plan.json`, `attribution-graph.json` |
| `select` | `perc-select` | `perc-selection.json` |
| `map` | `sae-feature-map` | `feature-map.json` |
| `evaluate` | `perc-components-plan`, `perc-components`, `perc-components-report` | `component-plan.json`, `component-ablation.jsonl`, `component-report.json` |

For GPU stages, a local HF directory with model files must exist. No `--allow-download` or `--trust-remote-code` flags are ever emitted. The runner checks the v0.12 graph preflight's forward count before launching its graph-estimation GPU process, and similarly checks the v0.10 component preflight before launching held-out GPU ablations. A stage is recorded as complete only after all of its outputs exist. Later stages re-hash the artifacts of earlier stages and refuse changed files. A completed stage can be requested again to **verify its hashes without rerunning it**.

If a stage is interrupted after writing partial output, the runner deliberately refuses to overwrite it. Inspect the partial data and logs before copying the entire pilot directory to an archive and restarting the affected stage in a *fresh pilot directory*. Run the offline planner again with the same parameters to reproduce the case file.

## Interpretation gates

A training graph with no measured source-to-sink path is a **negative diagnostic**. `select` stops rather than fabricating missing edges. A weak SAE reconstruction, zero feature activity, identical percolation/degree selectors, or intervention effects equal to norm-matched nulls should be recorded as weak or negative evidence. Train-derived candidates must remain frozen through the held-out ablation. The current readout scores fixed continuations and a first-token style logit; v0.7's blinded free-generation check is still needed to establish visible prose change.

The pilot budget is the **number of model-forward invocations per metered stage**. It is not a bound on wall time, token processing, VRAM, or cost. Run an initial small checkpoint on the 16 GB GPU, and record actual peak VRAM and device diagnostics outside the script. The current SAE corpus is extremely narrow; validation FVU on those scripted tokens does not establish SAE fidelity on independent, real-world text.

## Tests and code ownership

- `src/persona_holonomy/pilot.py` is standard-library orchestration. It does not compute attribution or patch model internals.
- `tests/test_pilot.py` checks offline plan determinism, split integrity, revision checks, run immutability, missing-stage failures, partial-output refusal and preflight budget aborts using a stub runner.
- The real-model acceptance gate remains the fully recorded Qwen run with pinned model/tokenizer, checkpoint hashes, FVU, held-out test cells and selector comparisons. For external replication, use new seed ranges and separately preregistered prompts.
