# PHB v0.11: train-only activation acquisition and TopK SAE fitting

**Status:** implementation and toy-model tests. There are no pretrained Qwen SAE or attribution-graph findings from this development environment. These commands address *SAE checkpoint acquisition*, not direct influence-graph estimation.

## Why this exists

PHB v0.10 accepts pinned SAE checkpoints and frozen PercInterp component selectors, but provided neither model-specific SAE weights nor an attribution graph. This release adds a reproducible path to collect decoder-block states from a local HF chat model, fit the exact [`percinterp.sae.TopKSAE`](https://github.com/MoralityLabAI/percolation-interp/blob/master/src/percinterp/sae.py) implementation, and produce a registry usable by the component-ablation code.

**Major distinction:** a small SAE trained only on PHB's narrow scripted persona prompts is an **engineering fixture / pilot**, not evidence of a stable, general-purpose SAE decomposition. A serious SAE study requires a larger, representative, independent activation corpus, stable feature coverage, held-out reconstruction fidelity, and replicate fits.

## Frozen membership

`phb sae-capture-plan` reuses PHB's causal `split_seeds` and `_partition_pairs` machinery. Given the default 8 seeded histories and two probes:

- 5 of the 8 seeds remain on PHB's training side; 3 are reserved for **future causal evaluation** and never passed to SAE fitting.
- Only `flaky_ci` is captured; `parser_validator` remains a held-out readout probe.
- Inside the 5 PHB training seeds, 4 contribute SAE optimization states and 1 contributes independent SAE validation states.
- Each captured seed contributes both `ordered` and `tail_matched` histories; data identity includes the full role-tagged prompt hash.

These memberships are deterministic for a fixed seed, independent of GPU choice. Even though the SAE validation seed is withheld from optimization, it is still inside PHB's **causal-training** partition. Do not report that validation FVU as performance on the PHB causal holdout.

```bash
phb generate --variation seeded_v1 --repetitions 8 --seed 100 \
  --conditions ordered,tail_matched --phases exit \
  --probes flaky_ci,parser_validator --output runs/sae-cases.jsonl
phb sae-capture-plan --input runs/sae-cases.jsonl \
  --heldout-probes parser_validator --output runs/sae-plan.json
```

Review the JSON and freeze model ID, checkpoint SHA/revision, layer list, cases, training hyperparameters and output paths **before** collecting any real data. `sae-capture-plan` never loads model weights.

## Capture: pretrained-model inference

Requires a **locally cached Hugging Face causal chat checkpoint** with compatible `model.layers`/`transformer.h` decoder blocks. GGUF/Ollama model names cannot be passed directly as HF paths. Default flags prevent model downloads and arbitrary model remote-code execution.

```bash
python -m pip install -e '.[test,mechanistic]'
phb sae-capture --model /path/to/CACHED_HF_QWEN \
  --revision ACTUAL_IMMUTABLE_HF_REVISION_OR_WEIGHTS_HASH \
  --input runs/sae-cases.jsonl --layers 0,mid,last \
  --heldout-probes parser_validator \
  --max-input-tokens 1536 --max-tokens-per-case 64 \
  --device cuda --dtype float16 \
  --output-dir runs/sae-capture-v1
```

The capture command runs full model forwards but retains at most 64 sampled token vectors per conversation and decoder block (including the final prompt token). It writes `float32` NumPy archives in `runs/sae-capture-v1/block-NNN.npz` with separate `train` and `validation` matrices and a SHA-256-pinned `capture-manifest.json`. No raw token IDs or prompt strings are exported to these archives. **Activation arrays remain potentially sensitive and should stay local.** A token sampling cap reduces output size, not model-forward compute, KV memory, or the full input-token count.

For an RTX 5080 with 16 GB VRAM, start with a small cached Qwen model at a precision that actually fits. If an input exceeds the token cap, the command raises rather than silently truncating it. The checkpoint revision is a caller declaration and is not automatically verified against loaded model bytes; log a trusted weights hash or pin a revision externally.

## Training: CPU/GPU after capture

Install the known PercInterp source at the pinned commit used by PHB CI, then run the trainer in a separate process. This avoids holding the model and three SAE optimizers in GPU memory simultaneously.

```bash
python -m pip install 'git+https://github.com/MoralityLabAI/percolation-interp@c0b3f6abf8e847292aaf1e21b8f4b804fd4d8ce0'
phb sae-fit --capture runs/sae-capture-v1/capture-manifest.json \
  --output-dir runs/sae-fits-v1 --n-latents 1024 \
  --k 16 --steps 200 --batch-size 32 --device cpu
```

This trains *one SAE per selected block* on frozen train-token states and evaluates held-out **SAE-validation seed** states without backpropagating through them. The emitted `sae-registry.json` records fit/validation FVU, latent coverage, checkpoint hashes, hidden sizes, training hyperparameters, input-capture digest, and split memberships. The checkpoints are weights-only local `.pt` files saved with the exact `TopKSAE.state()` structure consumed by PHB v0.10. Training RNG is seeded, although accelerators may introduce nondeterministic kernels.

SAEs trained on a few thousand persona-prompt tokens will tend to learn superficial prompt-specific features. Treat reconstruction fidelity as a minimum gate, and run larger independent-corpus training before attaching mechanistic conclusions.

## Registry + real attribution graph → frozen feature map

After estimating a **real directed feature attribution graph** from the same immutable model and training split, pass it to `phb perc-select` and freeze selectors. The selected feature IDs must use **`sae:<zero_based_decoder_block>:<latent_index>`** and declare `decoder_block` on each selected graph node.

```bash
# Graph edges must come from a validated direct-attribution estimator.
phb perc-select --graph runs/ACTUAL-direct-attribution.json \
  --output runs/perc-selection.json

phb sae-feature-map --registry runs/sae-fits-v1/sae-registry.json \
  --selection runs/perc-selection.json \
  --output runs/feature-map.json

phb perc-components-plan --manifest runs/perc-selection.json \
  --feature-map runs/feature-map.json --input runs/sae-cases.jsonl \
  --model /path/to/CACHED_HF_QWEN --heldout-probes parser_validator \
  --output runs/perc-components-plan.json
```

The feature-map exporter verifies model/tokenizer identity, exact PHB training seed and probe sets, selected latent bounds, graph block mappings, local checkpoint hashes, and existing v0.10 per-block selector budget rules. It **does not invent, interpolate, or relabel graph edge weights.** The source graph's method and validity remain independent obligations.

## Non-negotiable research checks

1. Report true held-out SAE validation FVU, latent coverage, reproducibility and sensitivity to initialization. Check that `TopKSAE` scale and decoding convention reproduce v0.10 contribution subtraction.
2. Estimate signed **direct** influence edges with independent causal checks; a static cosine similarity, correlated SAE firing, or a PHB log-likelihood score is not a direct edge.
3. Freeze selectors before examining PHB causal holdout results; compare equal per-block latent budgets, exact identity/null patches, independent seeds and disjoint questions.
4. Produce a negative finding when the percolation sieve matches the degree baseline, the reconstruction is poor, or no extra causal advantage appears.

**Remaining gap:** train a well-supported SAE on a representative external corpus and implement/validate a direct-attribution exporter for its latent graph. Then v0.10 can test PercInterp's *incremental* causal utility without substituting graph structure for evidence.
