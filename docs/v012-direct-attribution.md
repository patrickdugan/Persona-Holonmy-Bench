# PHB v0.12: signed, single-feature finite-difference attribution

**Implementation status:** executable and locally tested with a synthetic PyTorch decoder. Pretrained Qwen SAE checkpoints, attribution edges, and scientific results have **not** been obtained in this environment. No implicit weights download.

## Scientific question

Can a *training-only* graph of intervention-measured feature influences supply PercInterp with useful, signed, source-to-output edges, allowing a frozen percolation sieve to select features whose subsequent removal changes the held-out persona readout more than matched local selectors?

This closes the interchange-format and edge-estimation gap between the v0.11 SAE fit and v0.10 component ablation. It **does not** prove the percolation selector adds value or that a VPD holonomy mechanism is present.

## Estimator

For each paired `ordered`/`tail_matched` *training* history, capture the final prompt-token residual at selected ascending decoder blocks. Encode each state using its corresponding pinned PercInterp TopK SAE. Candidate feature indices in each block are chosen by mean absolute **training-only** activation contrast, never by held-out test performance:

\[
\Delta a_{\ell,i}=a_{\ell,i}^{\mathrm{ordered}}-a_{\ell,i}^{\mathrm{tail}},\quad
S_\ell=\operatorname{TopM}_{i}\mathbb E_{\mathrm{train}}|\Delta a_{\ell,i}|.
\]

At source block \(\ell\), remove exactly the **contrast contribution** of feature \(i\) at the last prefix token:

\[
h'_{\ell}=h_{\ell}-\frac{\Delta a_{\ell,i}}{s_\ell}W^{\mathrm{dec}}_{\ell,i,:}.
\]

Let \(\ell^+\) denote the **next sampled block**, even if intervening model blocks exist. The signed feature-to-feature influence is the observed finite difference

\[
w_{(\ell,i)\to(\ell^+,j)}=\mathbb E_{\mathrm{train}}
\left[a_{\ell^+,j}(h)-a_{\ell^+,j}(h'_{\ell})\right].
\]

At the last sampled block, the target is the first-token log-odds of the registered theatrical versus plain continuation, evaluated after the same prefix:

\[
m=\operatorname{logit}(t_{\mathrm{theatrical}})-\operatorname{logit}(t_{\mathrm{plain}}),
\quad w_{(L,i)\to\mathrm{output}}=\mathbb E_{\mathrm{train}}[m(h)-m(h'_L)].
\]

These are **direct, single-feature interventions on a specified hidden tensor**. Their response at the next sampled block includes all downstream computations between the two blocks. Effects from different features need not add up; interactions and SAE reconstruction error remain. This is *not* gradient-based Edge Attribution Patching (EAP), a complete causal circuit decomposition, or a claim about generated-text style. The graph input uses the explicit estimator label `single_feature_contrast_finite_difference.v1`.

### Stages and identifiers

- Nodes are `sae:<zero_based_decoder_block>:<latent_index>` and the sink is `output:first_token_style_logodds`.
- Each selected decoder block is its own consecutive **topological stage**. First-stage features are graph sources, even if they are model features. Other blocks are interior vertices and are eligible for PercInterp selection.
- Edges connect *adjacent sampled stages*; all signed weights are measured on training histories and aggregated as signed means. A zero-mean edge is omitted. Mean absolute response and the fraction of positive effects remain as diagnostics.
- The `provenance` object records training seed IDs, training probe IDs, the excluded test partition, model/tokenizer IDs, claimed revision, checkpoint SHA-256, case digest, and the exact edge estimator.

## Offline preflight and real run

Use the **same** `runs/seeded-cases.jsonl` for SAE collection and graph attribution. Obtain `sae-registry.json` by running `phb sae-fit` from the v0.11 protocol. At least **two distinct SAE block checkpoints** are necessary.

```bash
# Offline, checks hashes and train/test boundaries; does not load Qwen:
phb sae-attribution-plan \
  --input runs/seeded-cases.jsonl \
  --registry runs/qwen-saes/sae-registry.json \
  --model /path/to/local/HF-Qwen \
  --layers all --top-features 4 \
  --heldout-probes parser_validator \
  --output runs/attribution-plan.json

# Opt-in GPU step using a cached Hugging Face model:
phb sae-attribution \
  --input runs/seeded-cases.jsonl \
  --registry runs/qwen-saes/sae-registry.json \
  --model /path/to/local/HF-Qwen \
  --layers all --top-features 4 \
  --heldout-probes parser_validator \
  --device cuda --dtype float16 --max-total-tokens 1536 \
  --output runs/attribution-graph.json

# CPU graph-only selection, requiring the installed pinned PercInterp package:
phb perc-select --graph runs/attribution-graph.json \
  --shuffles 100 --random-controls 20 \
  --output runs/perc-selection.json

# Bind frozen selected feature IDs to the previously trained SAE files:
phb sae-feature-map --registry runs/qwen-saes/sae-registry.json \
  --selection runs/perc-selection.json \
  --output runs/feature-map.json

# Then follow docs/v010-component-ablation.md for held-out tests.
```

**Cost scaling:** for `N` training seed/probe pairs, and `M_l` selected candidates per sampled block, the conservative model forward count is `N × (2 + sum_l M_l)`. Each forward processes the prompt; the estimate is **not** a GPU-hour, latency, or VRAM estimate. Start with 4 candidates per block and 8 total seeds, not the full target dictionary.

## Falsification checks

1. Inspect SAE held-out validation FVU, active feature coverage, layer/tokenizer consistency, and identity hashes before trusting any attribution edges. A small scripted training corpus gives weak generalization evidence.
2. Verify train/test history and technical question disjointness, including the registry's test exclusions. Reject any attempt to tune source candidates on evaluated seeds.
3. Measure alpha-zero identity patches, zero-contrast features, and noise-matched controls. The current exporter provides signed local interventions; v0.10 evaluates **norm-matched orthogonal nulls on held-out cases**.
4. A graph without a nonzero source-to-sink path (flagged by `diagnostics.graph_status`), a sieve identical to degree, or a weakly reconstructed SAE is a valuable null/negative result. Do not convert it into a positive connectivity claim.
5. Record the exact weights revision and tokenizer *outside* the model path string. The metadata fields are exporter assertions, not cryptographic verification of an HF weight snapshot.
6. Replicate with independent training seeds, a larger corpus and another checkpoint. Compare first-token margin effects with blinded free generation. Multiple hypothesis correction and preregistration still apply to downstream tests.

## Implementation details

`src/persona_holonomy/attribution_graph.py` supplies the pure offline `graph_plan`, the PyTorch `measure_graph`, and the HF wrapper `graph_hf`. A `.pt` checkpoint is loaded as weights-only, or `.safetensors` by its safe reader. Optional model downloads and remote model code execution require explicit flags. The CPU PyTorch tests use synthetic checkpoints and a custom deterministic causal decoder and assert signed edges, strict topological ordering, reproducibility, and exclusion of held-out samples.
