# Current priority: v0.13 frozen Qwen workstation pilot

Read [docs/v013-workstation-pilot.md](docs/v013-workstation-pilot.md). Run `phb pilot-plan` to freeze the eight-seed training/held-out split and forward budgets without loading model weights. On the RTX 5080 with trusted cached HF Qwen weights, explicitly run `phb pilot-stage` in order: capture, fit, graph, select, map, evaluate. Inspect SAE validation FVU, graph paths, selector equivalence and paired held-out causal specificity after every stage. A missing graph path, altered artifact, absent checkpoint or exceeded model-forward budget is a stopping gate. No pretrained Qwen causal or holonomy result has been measured here. Fix compatibility bugs without altering the frozen test partition.

---

# Agent handoff: Qwen causal pilot (2026-10-22)

## Current priority: v0.11 SAE acquisition and direct-attribution gate

The repository now includes `src/persona_holonomy/sae_pipeline.py` and `docs/v011-sae-acquisition.md`. These close the **local SAE checkpoint acquisition** plumbing gap. Local tests pass; a pretrained Qwen SAE has not been trained in this development environment. PHB's causal holdout must remain untouched.

1. Review `phb sae-capture-plan --help` and generate seeded_v1 cases with `ordered,tail_matched`, `exit`, and `flaky_ci,parser_validator`. Run the offline plan first; inspect train/causal-test and SAE train/validation seed memberships.
2. On a workstation with a **locally cached trusted HF model**, run `phb sae-capture` with an explicit immutable `--revision`, chosen decoder blocks and bounded token counts. This writes train-only float32 activation archives, with SHA-256 hashes and no raw prompts. Keep the data local.
3. In a separate process, install the pinned PercInterp source and run `phb sae-fit`, producing actual `TopKSAE.state()` checkpoint files and reconstruction validation FVU. Inspect latent coverage and train/validation differences. For PHB-only scripted data, results are an engineering calibration with narrow coverage.
4. **Remaining high-priority scientific gate:** derive a properly validated, signed direct-attribution graph between identifiable SAE features, using the same pinned model and PHB train-only histories. Do not generate graph edges from cosine similarity, correlated firing, or invented sample weights. Follow PercInterp E03/E04 conventions where technically applicable; document estimator and reliability controls.
5. Freeze percolation and comparator selectors with `phb perc-select`; use `phb sae-feature-map` to bind `sae:<block>:<latent>` nodes to hash-checked SAE weights. Pass `phb perc-components-plan` before a held-out causal ablation. Record zero-effect/equal-budget selectors and negative findings.
6. Test code using `python -m pytest` plus GitHub's pinned PercInterp integration job; identify model revision, tokenizer, hardware, upstream commit, hashes, random seeds, FVU, token/case counts and exact commands. Submit targeted PRs and preserve compatibility with v0.10 component ablation.

**Scientific status:** no empirical pretrained-Qwen validation of persona transfer, SAE causal benefit, or percolation advantage has been reported. Further provenance and evidence are required before claiming model-level findings.

---

**Repository:** https://github.com/patrickdugan/Persona-Holonmy-Bench  
**Current software:** PHB v0.11.0, `main` (SAE acquisition implementation `02dde9a9f4204b7e15a9c2fb929bd0e390f348e8`).  
**Owner's next action:** Hand this repository to Claude Code or Codex on a Windows workstation with an RTX 5080 16 GB GPU.  
**Priority:** Run one scientifically interpretable *small* open-weight pilot before expanding the feature work.  
**License / data boundary:** MIT, public source and synthetic scripted conversation data. Keep credentials, proprietary transcripts, and potentially infohazardous material out of the repo.

## Research question and status

Does a conversation-conditioned, rank-one residual direction learned from ordered vs. tail-matched persona induction histories causally shift a held-out model's preference for theatrical versus ordinary continuations on *unseen technical questions*?

Versions 0.1–0.7 include transcript controls, activation capture, interventions, layer sweeps, seed-distinct histories, unseen-probe holdout, exact sign-flip inference, and offline cost preflight. The current code has been tested with mock responses and small PyTorch decoder fixtures. **No pretrained Qwen intervention result has been obtained yet.** The nickname “holonomy” is a hypothesis; transcript path dependence is insufficient to establish geometric holonomy, and the rank-one mean difference is not yet a VPD/SAE feature.

Read first: `README.md`, `docs/freegen-validation.md`, `docs/heldout-inference-protocol.md`, `docs/layer-sweep-protocol.md`, `docs/intervention-protocol.md`, and `docs/vpd-roadmap.md`.

## Work order

1. Inspect `git status` and `git log -3`; work on a feature branch and avoid overwriting any intervening commits. Review the actual `phb --help` and command-specific help, which take precedence over copied commands.
2. Verify Python, PyTorch, Transformers, CUDA, driver, and free VRAM. Run `python -m pip install -e '.[test,mechanistic]'` and `python -m pytest`. Record versions, GPU name, and any compatibility failures. Restrict downloads and model execution to trusted checkpoints.
3. Select an **already available local Hugging Face causal chat checkpoint** that fits 16 GB VRAM, preferably a small Qwen instruct model. The `phb sweep` HF runner expects decoder blocks and a chat template. Ollama/GGUF models cannot be passed as HF checkpoint paths. If downloading is required, obtain explicit permission before any large download.
4. Generate and audit cases, then run the offline preflight:
   ```powershell
   phb generate --variation seeded_v1 --repetitions 8 --seed 100 --conditions ordered,tail_matched --phases exit --probes flaky_ci,parser_validator --output runs/qwen-pilot-cases.jsonl
   phb sweep-plan --input runs/qwen-pilot-cases.jsonl --sources ordered --layers 0,mid,last --alphas 0,1 --heldout-probes parser_validator --output runs/qwen-pilot-plan.json
   ```
   Inspect the JSON plan for input digest, distinct training/test seeds, held-out probe separation, layers, and estimated forwards before using GPU.
5. Run the smallest supported CUDA sweep, using a path to the *actual cached HF checkpoint*:
   ```powershell
   phb sweep --model "C:\path\to\local\HF-chat-checkpoint" --input runs/qwen-pilot-cases.jsonl --sources ordered --layers 0,mid,last --alphas 0,1 --heldout-probes parser_validator --phases exit --device cuda --dtype float16 --max-total-tokens 1536 --output runs/qwen-pilot-sweep.jsonl
   phb sweep-report --input runs/qwen-pilot-sweep.jsonl --output-prefix runs/qwen-pilot-report
   phb sweep-plot --report runs/qwen-pilot-report.json --source ordered --phase exit --alpha 1 --output runs/qwen-pilot.svg
   ```
   If GPU OOM occurs, reduce `--max-total-tokens` only if the needed context remains intact, or choose a smaller checkpoint/shorter pilot. Record any truncation and never silently convert a failed experiment into a partial result. Check that `--layers` accepts `mid` on the installed build.
6. Check controls before interpreting effects: alpha=0 identity, self-transplant identity where applicable, seed and probe isolation, same-norm orthogonal null, and candidate/erasure sign orientation. Inspect first-token and later-token margins separately. Keep forced-continuation scores distinct from free-generation behavior.
7. Save a compact `runs/README.md` or `reports/qwen-pilot.md` with checkpoint name and immutable revision/hash, exact commands, source commit, data SHA-256, package versions, GPU/VRAM, token counts, runtime, exclusions, plots, nulls, uncertainties, and links to artifacts. Summarize negative/null results too. Avoid committing large weights or raw full activations.

## Acceptance gates

- Local tests pass; the same workflow passes CI after the PR.
- Preflight split is recorded and no train/eval prompt or seed leakage is detected.
- At least one *real pretrained checkpoint* produces saved baseline, matched null, candidate injection, and projection-erasure outcomes across preselected layers.
- Outputs include the first-token/later-token decomposition, effect sizes, paired seed-level statistics, and multiplicity disclosure.
- Run is reproducible from scripts and pinned checkpoint revision. Report **pilot only**, since an eight-seed split leaves too few held-out seeds for strong significance claims.

## After pilot

If the effect is measurable beyond the first token, preregister ~30 seed-varied histories, fresh technical probe families, a new validation split, and an independently scored free-generation test. If it is absent or null-matched, inspect transcript-token confounds, probe wording, prompt truncation, hook positions, and nonlinear/distributed directions before increasing GPU spend. Only after replication consider SAE dictionaries, VPD factorization, or a mathematically defined state-transport/holonomy experiment.

## Deliverables for the next agent

A small PR containing any necessary compatibility fixes, tests, a reproducible run configuration, the pilot report and figure (if obtained), and clear caveats. Link the exact GitHub Actions run. Make no claims about GPT-4.1 personality reactivation, consciousness, or model-internal holonomy from style scores.

**Suggested first message to Codex/Claude:** “Read `HANDOFF.md` and the PHB v0.6 protocols. Run tests and preflight first; then conduct the small cached-HF-Qwen CUDA pilot on my RTX 5080. Keep train/test probes disjoint, retain all null controls, write a reproducible report, and open a PR. Avoid GPU-heavy runs or model downloads until the preflight is reviewed.”


## v0.7 addition: free-generation validation (2026-10-22)

After the v0.6 Qwen forced-continuation layer sweep is verified, use a **preregistered layer** to test whether the same trained rank-one direction changes *the responses the model freely generates*. This protects against optimizing only a forced-log-likelihood readout. Version 0.7 adds:

- `phb freegen`: greedy response generation on unseen transcript seeds **and unseen probe questions**, with baseline / candidate / norm-matched orthogonal controls. The hook patches only the prefill call during KV-cached generation.
- `phb blind-export`: randomized rating CSV with anonymized sample IDs plus a separate private unblinding key.
- `phb blind-report`: all-ratings-complete enforcement and seed-clustered candidate-vs-orthogonal contrasts (style and answer quality, exact sign-flip p-values, Holm correction).

Use the **same cases, split, layer selection, and checkpoint revision** as the registered Qwen causal sweep. Do not pick a winning layer on the held-out data and then report the same holdout as confirmation. For a feasibility check:

```powershell
phb freegen --model "C:\path\to\local\HF-chat-checkpoint" --input runs/qwen-pilot-cases.jsonl --layer mid --alphas 1 --sources ordered --heldout-probes parser_validator --device cuda --dtype float16 --max-new-tokens 64 --max-total-tokens 1536 --output runs/qwen-pilot-freegen.jsonl
phb blind-export --input runs/qwen-pilot-freegen.jsonl --output-prefix runs/qwen-pilot-blind
# Have an independently blinded reviewer complete both numeric columns.
phb blind-report --ratings runs/qwen-pilot-blind-ratings.csv --key runs/qwen-pilot-blind-PRIVATE-key.json --output runs/qwen-pilot-blind-report.json
```

Read `docs/freegen-validation.md` for detailed ratings, controls, limitations and acceptance tests. Leave private rating keys out of PRs. If no pretrained weights are available, test the logic with fixtures, record that limit, and leave empirical claims pending. Existing toy-forward-hook tests cannot establish a model-level persona effect.


## PHB v0.8: PercInterp integration (2026-10-22)

Current implementation: [docs/percinterp-bridge.md](docs/percinterp-bridge.md). PHB now exposes `perc-select`, `perc-audit`, and `perc-holonomy`. Read the doc and tests first. Install `MoralityLabAI/percolation-interp` from its upstream repository on Python >=3.11 before running the real-library adapter tests.

- `perc-select`: Feed validated **direct** VPD/Goodfire component attribution edges into PercInterp's DAG sieve. Freeze the output and compare budget-matched local-degree and random alternatives. The included graph fixture is SYNTHETIC.
- `perc-audit`: Join frozen component-to-decoder-block mappings to a held-out PHB rank-one sweep. Its statistics concern **decoder blocks**, never individual component necessity. It rejects graph/sweep train-seed mismatch, repeated or missing causal control cells and unequal block budgets.
- `perc-holonomy`: From matched last-prefix-token hidden states, build rank-r PCA frames on (condition, phase) cells, use PercInterp's polar Procrustes transport, certify overlaps and loops, test gauge invariance, and evaluate a condition-label permutation null. Use seeded_v1 `peak,exit,washout_2` cases; fixed baseline prompts can yield rank-deficient frames.

**Open research obligations:** Run a real cached Qwen model on the RTX 5080, get real signed attribution edges from an independently checked source, freeze graph selectors on training-only histories, compare on held-out seeds/probes, and test whether individual selected SAE/VPD components (as opposed to whole blocks) show incremental causal effects versus local-degree and same-budget random controls. Report null outcomes. There are no pretrained-model findings from this crossover yet.

**First Codex/Claude instruction:** Verify v0.8 GitHub CI and `docs/percinterp-bridge.md`, run `tests/test_perc_bridge.py` and `tests/test_perc_holonomy.py` using the real `percinterp` installation, fix any adapter mismatch, then prioritize a cached-Qwen pilot. Avoid unrequested model downloads or expensive GPU jobs.


## 2026-10-22: v0.9 PercInterp robustness follow-up

Latest work is the provenance-checked PHB × PercInterp bridge. See `docs/v09-percinterp-robustness.md` and `docs/percinterp-bridge.md`.

- For `phb perc-audit`, exporter assertions `provenance.model`, `provenance.train_seeds`, and `provenance.training_probe_ids` must match the sweep's declared model and exact training split. Optional `provenance.tokenizer` also must match. Freeze original files and immutable checkpoint revisions for real reproducibility.
- Two component selectors that map to the same selected decoder-block set receive equivalent layer-level interventions; the audit explicitly flags this. Do not treat their original component selection difference as independently validated.
- `phb perc-holonomy` uses whole-seed condition-label permutations shared across all probes and phases. The number of randomization units is distinct seeds, not seed×probe×phase.
- CI pins PercInterp commit `c0b3f6abf8e847292aaf1e21b8f4b804fd4d8ce0`; the earlier v0.8 mechanistic CI failure was caused by a PyTorch tensor-versus-pytest.approx comparison, and a scalar test fix is included here.
- Next priority remains a pinned open-weight checkpoint experiment on local GPU, first as a preflight pilot. Then estimate direct component-attribution edges on a *training-only* partition, freeze PercInterp selectors, and assess results on held-out examples. Component-level causal patching is future work.

Report negative effects and selectors that coincide with local degree. Never interpret a block-level audit as a component-level causal finding or label percolation-frame transport as endogenous model state transport.

## v0.10: component-level PercInterp causal test

The next research step is described in [docs/v010-component-ablation.md](docs/v010-component-ablation.md). PHB now provides `phb perc-components-plan`, `phb perc-components`, and `phb perc-components-report`. These test a **frozen, equal-per-block-budget PercInterp component selection** through pretrained TopK SAE decoder-contribution subtraction at the final prompt token. All selectors receive norm-matched random-orthogonal controls. Report comparisons by held-out **seed** after holding out an entire technical question family.

Before running this on Qwen, obtain genuine pretrained SAEs for the **same model/revision**, derive faithful direct-attribution graph edges using only training seeds/probes, freeze `phb perc-select`, construct the explicit feature-map JSON, pin checkpoint SHA-256 values, and confirm the graph's `train_seeds` and `training_probe_ids` match PHB's deterministic split. Begin with `phb perc-components-plan`, which loads no model weights and fails on missing hashes, provenance mismatches, or unequal per-block latent budgets. A toy decoder and TopK checkpoint path pass CPU tests; real Qwen component effects remain unmeasured. If attribution graphs or trained SAEs are unavailable, prioritize their construction and validation, not generating arbitrary latent mappings.


## v0.12: training-only signed single-feature attribution graph (October 22, 2026)

Software v0.12 adds `phb sae-attribution-plan` and `phb sae-attribution`. Read `docs/v012-direct-attribution.md` first. The exporter loads locally hashed PercInterp TopK SAE checkpoints from the v0.11 registry, selects candidate latents by absolute *training-only* ordered-vs-tail-matched activation contrast, and estimates signed, finite-difference effects on later SAE latents or the first-token theatrical/plain style log-odds. It emits `phb.percinterp.attribution.v1` graph JSON for `phb perc-select`, `phb sae-feature-map`, and the v0.10 held-out `phb perc-components` comparison.

Priorities for the next coding agent:
1. Run `python -m pytest`, inspect model/SAE architecture and the v0.12 CI job; fix compatibility issues on a branch and PR.
2. Generate cases with at least two probe IDs and disjoint seeds, capture/fix SAEs on training only; review validation FVU and checkpoint integrity.
3. Run `phb sae-attribution-plan --input runs/cases.jsonl --registry runs/saes/sae-registry.json --model /path/to/CACHED_QWEN --top-features 4` without loading model weights. Audit exact forward-count budget.
4. If the local 16GB VRAM model fits, run `phb sae-attribution` at four latents per SAE block, then `phb perc-select`. Examine `diagnostics.graph_status`, path availability, edge cancellation, and the measured first-token metric. Declare nulls.
5. Freeze selector outputs before running feature-map export and held-out per-component ablation; compare same-budget degree and random selectors. Validate with fresh generation and blinded assessment.

**Scientific boundary:** These edges are causal *local finite differences* for a specific final-prefix-token tensor and continuation metric, not additive EAP gradients, model-wide causal attribution, or proof of geometric holonomy. No pretrained Qwen graph result has been obtained here. The model revision in metadata is an exporter assertion; independently pin the actual weight SHA/revision.
