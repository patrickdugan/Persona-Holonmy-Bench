# PHB v0.10: PercInterp selection → TopK SAE feature-contribution ablation

**Status:** implemented, tested on synthetic PyTorch decoders. No pretrained Qwen effects measured. This is an optional GPU experiment and uses only locally provided checkpoints by default.

## Research question

Does the source-to-output component backbone chosen by [PercInterp](https://github.com/MoralityLabAI/percolation-interp) predict a greater *causal decrease in the theatrical style margin* after selected SAE latent contributions are subtracted than equal-budget local-degree, total-attribution, and random component selections?

This extends the prior v0.9 **decoder-block** audit to **specific decoder feature indices**. A latent is a trained SAE feature index, not a transformer neuron, VPD parameter factor, or human-defined semantic concept. A valid causal effect on a chosen tensor and readout does not by itself prove the graph's source attribution was faithful.

## Data contracts

1. `phb perc-select` produces a **frozen selection manifest** (`phb.percinterp.selection.v1`) from directed *direct-attribution edges* estimated using training data. Its `selected_nodes` is the candidate/controls registry and its `provenance` must declare the model, tokenizer (if used), training seeds and training probe IDs.
2. A separate **feature map** (`phb.percinterp.feature_map.v1`) resolves selected graph node IDs to `(zero-based decoder block, SAE latent index)` and supplies one frozen [PercInterp TopKSAE](https://github.com/MoralityLabAI/percolation-interp/blob/master/src/percinterp/sae.py) checkpoint per selected block.
3. The exported PHB cases (`seeded_v1`) contain disjoint train/evaluation history seeds and technical probe families. Graph training seeds and probe IDs must match the case split exactly.

Example feature map (paths relative to the map JSON file):

```json
{
  "schema": "phb.percinterp.feature_map.v1",
  "model": "/models/my-local-qwen",
  "tokenizer": "/models/my-local-qwen",
  "selection_graph_sha256": "COPY_exact_graph_sha256_FROM_perc_select",
  "checkpoint_revision": "PINNED_QWEN_REVISION_OR_WEIGHTS_SHA256",
  "checkpoints": {
    "8": {"path": "saes/08.safetensors", "top_k": 32, "sha256": "LOWERCASE_CHECKPOINT_FILE_SHA256"},
    "16": {"path": "saes/16.safetensors", "top_k": 32, "sha256": "LOWERCASE_CHECKPOINT_FILE_SHA256"}
  },
  "features": {
    "component:layer8:25": {"block": 8, "latent": 25},
    "component:layer16:55": {"block": 16, "latent": 55}
  }
}
```

The `features` object must cover **all** the selected node IDs, including random controls. Each feature location must be unique and agree with the selection manifest's `node_to_decoder_block`. All selectors must choose equal numbers of latents **in each decoder block**, rather than just an equal total count. If an upstream PercInterp selection is matched only by topological stage and yields unequal per-block budgets, **the component test rejects it**. Freeze a scientifically justified balanced selection upstream rather than silently filling slots based on evaluation data.

The checkpoints use either safe tensor files (`.safetensors`) containing `W_enc`, `W_dec`, `b_enc`, `b_dec`, `scale`, or locally trusted `.pt` checkpoints saved from `TopKSAE.state()`, loaded with `torch.load(weights_only=True)`. For `.pt`, the embedded `config.k` must match `top_k`. No remote checkpoint URLs are allowed. `scale` must be finite and positive; dictionary sizes must match the target decoder block. Checkpoint file SHA-256 is verified at preflight and again at execution.

## The intervention

At each held-out **ordered** conversation history, run one unmodified forward pass, saving the *final prompt token* activation `h_ℓ` at each selected decoder block. The [PercInterp TopKSAE](https://github.com/MoralityLabAI/percolation-interp/blob/master/src/percinterp/sae.py) encoding is

\[
 a_\ell = \operatorname{TopK}_{k_\ell}\!\left(\operatorname{ReLU}\left[(s_\ell h_\ell-b_{dec,\ell})W_{enc,\ell}+b_{enc,\ell}\right]\right).
\]

For a frozen selected set of feature indices `S_ℓ`, the feature contribution is

\[
 \Delta_\ell(S_\ell) = \frac{1}{s_\ell}\sum_{j\in S_\ell}a_{\ell,j}W_{dec,\ell,j,:}.
\]

For the *same held-out prompt*, insert a decoder-block forward hook and patch

\[
 h'_\ell = h_\ell - \alpha \Delta_\ell(S_\ell),\quad 0<\alpha\leq 1.
\]

The `Δ` is computed on the **unmodified baseline** and frozen through the intervention, even when multiple blocks are patched. This preserves the original SAE reconstruction error, and makes the intervention magnitude/feature identity reproducible independently of upstream perturbations. It is a precise intervention on selected *predicted contributions*, **not** a full causal counterfactual in which SAE encoders are rerun on every subsequently modified layer state. TopK zero-valued features cause zero patch effects.

For each decoder block separately, create a seeded random vector orthogonal to `-Δ` with identical L2 norm. Apply these null patches at the same locations and with the same α. If `Δ` has zero norm, both candidate and null are zero. Also score the unmodified baseline and an α=0 sham. Compare two fixed continuations with the same technical content and different presentation styles using PHB's forced-continuation margin. Record first-token and later-token margins separately.

For each held-out seed/probe case, define

\[
 m = \overline{\log p(\text{theatrical})} - \overline{\log p(\text{plain})}, \qquad
 z = (m_{baseline}-m_{candidate})-(m_{baseline}-m_{null}) = m_{null}-m_{candidate}.
\]

Positive `z` indicates that the selected feature-subtraction reduced theatrical preference **more** than the norm-matched orthogonal perturbation. Report both raw candidate effect and null effect. A comparison of the sieve against control selector `C` uses the held-out seed-average difference `z_sieve - z_C`, with exact two-sided sign-flip inference on **seeds**, then Holm correction over preregistered controls. Identical selected feature sets must give identical results. Five- or eight-seed trials are engineering shakedowns; with three held-out seeds the smallest two-sided exact p-value is 0.25.

## Preflight then actual model run

Generate cases across at least two technical probes, match the graph's frozen training split, and prepare a feature map with actual SAE checkpoints. Start with a small checkpoint before scaling up:

```bash
phb generate --variation seeded_v1 --repetitions 8 --seed 100 \
  --conditions ordered,tail_matched --phases exit \
  --probes flaky_ci,parser_validator --output runs/component-cases.jsonl

# CPU metadata and checkpoint hashing; no model weights loaded.
phb perc-components-plan \
  --manifest runs/perc-selection.json \
  --feature-map runs/feature-map.json \
  --input runs/component-cases.jsonl \
  --model /models/my-local-qwen \
  --heldout-probes parser_validator \
  --output runs/component-plan.json

# Locally cached HF chat weights, no automatic model download.
phb perc-components \
  --manifest runs/perc-selection.json \
  --feature-map runs/feature-map.json \
  --input runs/component-cases.jsonl \
  --model /models/my-local-qwen \
  --heldout-probes parser_validator \
  --alpha 1 --device cuda --dtype float16 \
  --output runs/component-ablation.jsonl

phb perc-components-report --input runs/component-ablation.jsonl \
  --output runs/component-report.json
```

The commands fail before expensive model execution on invalid graph/checkpoint hashes, inadequate provenance, duplicate feature locations, unequal per-block budgets, or seed/probe overlap. The offline plan emits the test split, exact selector budgets, source hashes and estimated model forward calls. It does **not** estimate peak VRAM or inference wall time.

## Acceptance gates for a research claim

- Verify the SAE against its source implementation, training data and reconstruction fidelity (FVU). Report sparsity, active-feature coverage, encoder conventions and feature stability across seeds. For models larger than the checkpoint training model, do not reuse the SAE.
- Supply actual **direct-attribution** graph edges with experiment-level provenance, freeze all component selectors before inspecting held-out outcomes, and compare only equal-budget selectors.
- Verify the alpha-zero sham, same-norm null at every decoder block, fixed recipient prompt, complete seed/probe cells, checkpoint SHA and identical-feature-selection invariance.
- Report first-token and later-token effects separately, plus a blind free-generation check using PHB v0.7. Prefer fresh seed/probe replication with multiple independent checkpoints.
- If selection equivalence, weak SAE reconstruction, or absence of an incremental benefit explains the outcome, mark it as a negative or inconclusive percolation result. This gate is intended to **falsify** an incremental benefit, not guarantee one.

The v0.10 fixture proves that the feature map, checkpoint loader, hooks, random controls and seed-clustered report operate on a small PyTorch decoder. It does not evaluate real model feature importance.
