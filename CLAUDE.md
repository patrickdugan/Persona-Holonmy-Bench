# Claude Code handoff: Persona Holonomy Bench

Updated 2026-10-23. This is the **entrypoint for Claude Code**. See [HANDOFF.md](HANDOFF.md) for historical context and [docs/v013-workstation-pilot.md](docs/v013-workstation-pilot.md) for the executable pilot procedure.

## Objective and exact current status

Build a reproducible, falsifiable test of conversation-conditioned persona persistence and whether PercInterp-selected SAE features predict held-out causal effects better than equally budgeted degree/attribution/random selectors.

Current repository: `patrickdugan/Persona-Holonmy-Bench` (the repository name is intentionally spelled **Holonmy**). Baseline v0.13 commit: `4482ae41b05f6e5ffed712dca426e080c35c8b6d`. GitHub Actions run `39814548641` completed **successfully** (verified 2026-10-23). The tested implementation provides a six-stage pilot: capture → fit → graph → select → map → evaluate. Earlier v0.7 provides separate free-generation/blinded-response validation.

**Empirical status:** toy-decoder and synthetic integration checks are passing. There is **no pretrained-Qwen SAE training, signed graph, component-ablation result, or measured holonomy result** from this project yet. Do not present synthetic tests as model-science findings.

## First actions for Claude

1. Inspect current `git status`, branch, `git log -5`, and the latest CI workflow. Preserve work from any subsequent commits; use a feature branch and submit a PR for code changes.
2. Read `docs/v013-workstation-pilot.md`, `docs/v012-direct-attribution.md`, `docs/v011-sae-acquisition.md`, `docs/v010-component-ablation.md`, and `docs/percinterp-bridge.md`. Confirm the installed CLI help rather than guessing flags.
3. Run `python -m pip install -e '.[test,mechanistic]'` and `python -m pytest`. For PercInterp tests use **Python 3.11+** and the exact PercInterp commit pinned in `.github/workflows/ci.yml`, not an unpinned installation. Record environment versions.
4. Check that the Windows RTX 5080 **16 GB** workstation has a trusted, local Hugging Face causal-chat checkpoint compatible with PyTorch CUDA, Transformers decoder-block hooks, and chat templating. A GGUF or Ollama model label is not an HF model directory. Confirm available VRAM, CUDA functionality, immutable weights revision, tokenizer identity. Ask before a substantial checkpoint download, remote-code execution, or expensive model run.
5. Run the offline plan using the same real checkpoint identity across every stage:

```powershell
phb pilot-plan `
  --model 'C:\models\qwen-local-hf' `
  --revision 'PASTE_REAL_40_HEX_COMMIT_OR_64_HEX_WEIGHT_DIGEST' `
  --output-dir 'runs\qwen-pilot-001' `
  --repetitions 8 --layers mid,last `
  --top-features 4 --random-controls 4 `
  --max-total-tokens 1536 --max-tokens-per-case 32 `
  --max-model-forwards 250
```

Inspect `plan.json` and the SHA-256s, then execute **one stage at a time** with explicit resource review:

```powershell
$P = 'runs\qwen-pilot-001\plan.json'
phb pilot-stage --plan $P --stage capture
phb pilot-stage --plan $P --stage fit
phb pilot-stage --plan $P --stage graph
phb pilot-stage --plan $P --stage select
phb pilot-stage --plan $P --stage map
phb pilot-stage --plan $P --stage evaluate
```

No implicit model download or `--trust-remote-code` is permitted in the pilot runner. If a stage stops for a failed scientific gate or corrupted/interrupted artifact, document and debug the cause. Preserve the original output and start a fresh pilot directory rather than overwriting hashes or bypassing guards.

## Experimental non-negotiables

- Keep **train and held-out seeds disjoint** and hold out an entire technical probe family. Default eight-seed run has five train and three held-out seeds, one of the five reserved for SAE validation. Three independent held-out seeds cannot yield a two-sided exact sign-flip p-value below 0.25; treat this as a **shakedown only**.
- Train SAE dictionaries using training-only activations. Report separate validation FVU, active-feature coverage, dead latents, tokenizer and model revision, token counts, and checkpoint SHA-256. Weak reconstruction quality is a research result and a gate.
- Construct signed direct-attribution edges from the specified **single-feature finite-difference interventions**. Do not substitute correlation, cosine similarity, or fabricated attribution values; record the exact estimator, intervention position, graph completeness, and diagnostics.
- Freeze PercInterp's sieve, degree, total-attribution and random selections **before** inspecting held-out causal effects. Match SAE latent budgets per decoder block. If the graph has no source-to-output path or the selectors coincide, report it rather than force a favorable result.
- Preserve alpha-zero identity, norm-matched orthogonal nulls, identical held-out test cells and frozen checkpoint hashes. Separate first-token from later-token effects. Report raw negative/null effects.
- The current forced-continuation style score is a **proxy**. Follow it with preregistered free-generation comparisons and blinded style plus factual-quality ratings if the pilot warrants extension.
- State clearly that transcript-conditioned path dependence is not proof of **geometric holonomy**, SAE factors are not automatically VPD factors, and an observed rank-one direction does not demonstrate any GPT-4.1 persona inside the model.
- This is public MIT research. Do not upload confidential data, credentials, model weights, large raw activation archives, private rating keys, or sensitive material into the public repo.

## Acceptance / final deliverable

Provide a PR or report linked to the exact code commit and passing CI run, containing the pinned configuration, plan/input hashes, seed/probe split, run commands, environment and GPU peak-memory notes, SAE fit diagnostics, signed-graph provenance, frozen selector sets, paired held-out effects and controls, uncorrected/corrected statistics, exclusions and limitations. Preserve failing/negative outcomes. Document where the pipeline stopped if actual cached weights or sufficient VRAM are unavailable.

Suggested summary for the owner: `Qwen checkpoint/revision: …; stages completed: …; CI: …; science gates: …; evidence: …; next action: …`.
