# PHB × PercInterp: attribution sieves, held-out causal gates, and frame holonomy

**Status: v0.9 optional integration with provenance and seed-cluster controls.** Both repositories are MIT-licensed. The Python distribution/import for [PercInterp](https://github.com/MoralityLabAI/percolation-interp) is `percinterp`. The upstream package requires Python 3.11+ and NumPy/SciPy; PHB's core continues to work under Python 3.10 without those dependencies.

This is intentionally **two separate experiments**. The direct-attribution path selects candidate components from an *externally calculated* layered influence graph. The frame-transport path tests a particular mathematical connection on PHB residual activation samples. Neither input can be quietly substituted for the other.

## Setup

```bash
python -m pip install -e '.[test]'
python -m pip install 'git+https://github.com/MoralityLabAI/percolation-interp'
python -m pytest tests/test_perc_bridge.py tests/test_perc_holonomy.py
```

`percinterp` is loaded only when a PercInterp command runs. Clone and install from the known repo rather than assuming a PyPI package with the same name is authentic. For precise replication, pin the upstream git commit in the environment manifest.

## A. Direct attribution graph → sieve → frozen component selection

Export a JSON file with the schema below. Each edge is a **signed direct attribution** from an actual EAP, VPD component attribution, or another validated estimator. Magnitudes are used by the sieve, and the original signs are preserved in the input. Raw layer-state cosine similarities and the PHB rank-one intervention scores are **not** acceptable substitutes for direct edges. If a Goodfire `ComponentKey.encoded()` ID exists, use that as the exact node ID.

```json
{
  "schema": "phb.percinterp.attribution.v1",
  "provenance": {
    "edge_semantics": "direct_attribution",
    "estimator": "YOUR_VERIFIED_EAP_METHOD",
    "model": "EXACT_MODEL_REVISION",
    "train_seeds": [100, 101, 102],
    "training_probe_ids": ["flaky_ci"],
    "tokenizer": "OPTIONAL_TOKENIZER_ID"
  },
  "nodes": [
    {"id": "input:token", "stage": 0},
    {"id": "component:layer0:12", "stage": 1, "decoder_block": 0},
    {"id": "component:layer1:5", "stage": 2, "decoder_block": 1},
    {"id": "output:style_margin", "stage": 3}
  ],
  "edges": [
    {"source": "input:token", "target": "component:layer0:12", "signed_weight": 0.7},
    {"source": "component:layer0:12", "target": "component:layer1:5", "signed_weight": -0.4},
    {"source": "component:layer1:5", "target": "output:style_margin", "signed_weight": 0.6}
  ],
  "sources": ["input:token"],
  "sinks": ["output:style_margin"]
}
```

`stage` is a **topological order** (zero-based, consecutive), not necessarily the same as the decoder block. A `decoder_block` is an **explicit** mapping required only for the weaker layer-level causal audit. Multi-component mappings to one decoder block are allowed for selection, but the audit refuses comparisons with unequal unique-block budgets.

```bash
# This included fixture is SYNTHETIC, for pipeline smoke tests only.
phb perc-select \
  --graph examples/percinterp/direct-attribution-SYNTHETIC.json \
  --shuffles 32 --random-controls 12 --seed 20261022 \
  --output runs/perc-SYNTHETIC-selection.json
```

`perc-select` calls PercInterp's `from_attribution_edges`, `rank_entry`, `profile`, `sieve` and `degree_entry`. It records a graph fingerprint, percolation occupation, backbone node IDs, source-to-sink span threshold, stage quotas, shuffle p-value, and budget-matched local-degree, total incident-weight, and random baselines. If the graph-only connectivity peak selects an empty interior backbone, it explicitly records a **spanning-threshold fallback**. A weak or insignificant graph is still just an exploratory candidate selector.

### Join to PHB held-out experiments

Prepare a causal `phb sweep` with all decoder blocks named in **every** frozen selector. Ensure the attribution source was built from the same training seeds and from no held-out inputs. Then:

```bash
phb perc-audit \
  --manifest runs/perc-selection.json \
  --sweep runs/qwen-sweep.jsonl \
  --source ordered --phase exit --mode inject --alpha 1 \
  --output runs/perc-layer-audit.json
```

The audit refuses overlapping train/test seeds, mismatched training-probe partitions, **different model checkpoint identifiers**, tokenizer discrepancies when the optional field is present, missing blocks, missing orthogonal controls, unequal selector block budgets, and incomplete matched seed/probe cells. **`provenance.training_probe_ids` and `provenance.model` are required for an audit**, while graph-only selection can still run without training-probe metadata. Metadata is an exporter assertion rather than an independently verified provenance chain. The selection's graph digest makes later mutation detectable if the original graph is retained. It calculates within-seed per-block candidate-minus-orthogonal specificity, then paired sieve-versus-control contrasts with sign-flip p-values and Holm correction.

**Critical boundary:** PHB's current interventions change a *whole decoder-block residual direction*. A successful layer audit can suggest which blocks merit expensive component-level tests; it cannot establish causal necessity or sufficiency of an individual PercInterp-selected component. Component-level activation patching must be implemented before making that stronger claim. A percolation selector identical to a local-degree selector has no demonstrated incremental advantage.

## B. PHB activations → genuine, defined discrete frame transport

PHB already saves full last-prefix-token vectors at chosen hidden-state indices. We can define a graph of experimental states using two conditions across successive checkpoints, estimate orthonormal PCA frames from matched seed/probe variation at *the same hidden-state index*, and apply the precise discrete connection used by PercInterp E08:

\[Q_{u\to v} = \operatorname{polar}(U_v^\top U_u).\]

Edges connect adjacent phases within each condition and corresponding phases across conditions. The two routes through each square define cycles. PercInterp's `frame_edge_transports` and `analyze_connection` return certified cycles, Betti curves, phase onsets, eigenphases, and ordered loop transports. A local basis-change test rotates every frame independently and verifies that loop spectral invariants remain unchanged.

```bash
phb generate --variation seeded_v1 --repetitions 12 --seed 100 \
  --conditions ordered,tail_matched \
  --phases peak,exit,washout_2 \
  --probes flaky_ci,parser_validator \
  --output runs/perc-frame-cases.jsonl

# Requires a real locally cached model, an explicitly chosen checkpoint, and GPU/CPU time.
phb capture --model /path/to/local/HF-Qwen \
  --input runs/perc-frame-cases.jsonl --layers 8 \
  --output runs/perc-frame-activations.jsonl

# Subsequent analysis is CPU-only.
phb perc-holonomy --activations runs/perc-frame-activations.jsonl \
  --layer hs_8 --conditions ordered,tail_matched \
  --phases peak,exit,washout_2 --rank 2 --min-overlap 0.1 \
  --null-shuffles 99 \
  --output runs/perc-holonomy.json
```

Why start at `peak`? The seeded-v1 baseline prompt is identical across seeds for a given probe, which can make baseline PCA **rank deficient**. Peak and later phases include seeded content variants. The command rejects unmatched seed/probe samples, mixed checkpoint identities, degenerate frames, and weakly supported links. Gauge invariance is a necessary implementation check; it does not demonstrate physical significance. The adapter includes a condition-label permutation null: it swaps ordered/tail labels **once per entire history seed, consistently across all probes and phases**, then refits frames and recomputes maximum certified loop distances. The exchangeability unit is a seed, so correlated probes never artificially increase null degrees of freedom. These are within-dataset nulls. Independent new seeds and another model family remain necessary for research conclusions.

## Working hypothesis and decision gate

1. Use PercInterp's source-to-sink sieve to identify an attribution backbone. Compare it with local degree and budget-matched random selectors, then freeze the candidate IDs **before** evaluating causal outcomes.
2. Check whether selected decoder blocks transfer stylistic effects to withheld probes using PHB's existing causal sweeps and free-generation ratings. Treat this as a screen for more precise component interventions.
3. Examine whether the persona-induction trajectory exhibits a certified, repeatable PCA-frame holonomy signal above shuffled-history nulls. Report loops that fail overlap certification and null results.
4. If layer-level audits provide no improvement over degree, or geometric phase invariants are unstable under replication, record the negative finding instead of interpreting topological structure as persona persistence.

The two experiments can be connected in a later study by computing **attribution edges between identifiable SAE/VPD components** under PHB induction states, then patching the exact selected features on held-out histories. That is the step that would substantiate the proposed percolation-to-causal bridge.
