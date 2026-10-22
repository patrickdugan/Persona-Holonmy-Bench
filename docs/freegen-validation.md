# v0.7: held-out free-generation validation and blinded ratings

## Why this exists

A rank-one residual intervention that changes a *forced continuation log-likelihood margin* may fail to change what the model actually writes. The v0.6 sweep uses two fixed alternative answers with different wording and token counts. v0.7 adds a separate, pre-registered behavioral readout: greedy model generations under an unmodified baseline, the trained intervention, and an equal-norm orthogonal control. A blinded rater independently scores the generated prose.

This is a *validation* stage. A persuasive result needs both a reliable causal forced-score signal and a held-out, independently scored free-generation effect. Neither is equivalent to identifying an SAE/VPD feature, a persistent internal agent, or geometric holonomy.

## Reproducible GPU workflow (16 GB card: start small)

Use the **same input cases, seed split, source, reference, probe holdout, and proposed layer** as the previously registered forced-score test. Pick one layer and one intervention strength *before* reviewing free-generation responses. Model downloads and remote code execution are disabled unless explicitly opted in.

```bash
python -m pip install -e '.[test,mechanistic]'
python -m pytest

# Preserves distinct histories across seeds and holds out a different question family.
phb generate --variation seeded_v1 --repetitions 8 --seed 100 \
  --conditions ordered,tail_matched --phases exit \
  --probes flaky_ci,parser_validator \
  --output runs/pilot-cases.jsonl

phb sweep-plan --input runs/pilot-cases.jsonl --sources ordered \
  --layers 0,mid,last --alphas 0,1 --heldout-probes parser_validator \
  --output runs/pilot-plan.json

# A local HF causal LM with its chat tokenizer, not an Ollama or GGUF filename.
phb freegen --model /path/to/cached/HF-chat-model \
  --input runs/pilot-cases.jsonl --layer mid --alphas 1 \
  --sources ordered --reference tail_matched --phases exit \
  --heldout-probes parser_validator --split-seed 20261022 \
  --max-new-tokens 64 --max-total-tokens 1536 \
  --device cuda --dtype float16 --output runs/pilot-freegen.jsonl

phb blind-export --input runs/pilot-freegen.jsonl \
  --output-prefix runs/pilot-blind --blind-seed 1729
```

The resulting files are `runs/pilot-blind-ratings.csv` and `runs/pilot-blind-PRIVATE-key.json`. **Keep the key private from the rater until every rating is complete.** The CSV contains `sample_id`, the neutral technical question, the generated answer, and two empty rating columns. The case/control identifiers are absent. Present responses in their shuffled order.

Score **style_rating_0_to_4** as 0 = conventional technical prose, 2 = partly stylized, 4 = strongly theatrical/game-quest language. Score **answer_quality_0_to_4** independently as 0 = useless/incorrect, 4 = correct and directly useful. Avoid looking at model condition, ordering, project labels, or key. Complete every row, including duplicate-looking outputs; maintain identical rating guidance across raters.

When completed:

```bash
phb blind-report --ratings runs/pilot-blind-ratings.csv \
  --key runs/pilot-blind-PRIVATE-key.json \
  --output runs/pilot-blind-report.json
```

The report groups scores **by held-out transcript seed** and calculates candidate-minus-orthogonal specificity with the sign reversed for erasure. It separately reports change relative to the baseline, and answer-quality difference. The seed-clustered two-sided sign-flip p-value and Holm adjustment cover every reported nonzero-dose source/layer/mode/phase cell. `blind-report` rejects incomplete rating sheets and unmatched samples.

**Important:** with eight total seeds and a 40% test fraction, three held-out seeds yield a minimum attainable two-sided exact p-value of 0.25. This is a feasibility test, not a significance experiment. Use a registered, sufficiently large *new* seed range for confirmation after the pilot. Avoid claiming a positive model effect based on the mocked test decoder.

## Mechanistic details and caveats

- The hook modifies only the final prefix position on the first (prefill) forward call during `generate`. Subsequent incremental generation calls are unmodified. The patched prefix can influence later tokens indirectly through downstream representations/attention; at the final decoder block, later-token effects may be very weak.
- Generation uses greedy decoding (`do_sample=False`, `num_beams=1`) to pair identical prefixes; this is a narrow deterministic operational definition of behavior. Repeat with preregistered sampled decoding and independent random seeds in future versions.
- The activation direction is estimated only on training histories and training probes. Evaluation uses unseen histories and a held-out technical question, via the same split logic as v0.6. It is not guaranteed semantic out-of-distribution transfer.
- The orthogonal perturbation is per-target and has the same L2 norm as the candidate. Equal norm cannot guarantee equal semantic or distributional disruption. A null test still needs careful interpretation.
- Model outputs and human ratings are not independent across probe texts within a transcript seed. Cluster and permutation-test at the **seed** level, never at the token or sample level.
- Record actual model revision, tokenizer, quantization, device, Python/Torch/Transformers versions, prompt and results hashes, intervention layer, decoding configuration, runtime, and any errors in the pilot report. The JSONL includes case IDs and seed partition information; the blind CSV does not.
- Leave the private key out of public pull requests. `runs/` and `*-PRIVATE-key.json` are ignored by Git.
- A single rater's ratings are subjective. For a confirmatory study, use multiple independently blinded raters and report disagreement/inter-rater consistency. The current reporter accepts one completed rating sheet and does not yet model rater effects.
- Correctness should be evaluated separately. A style shift accompanied by lower answer quality is not an unambiguous performance improvement.

## Acceptance checks

1. `python -m pytest` passes, including CPU decoder-hook simulations that invoke the block once for prefill and again for cached one-token decoding.
2. `phb sweep-plan` succeeds and identifies the predetermined history/question split; compare it with the recorded freegen partition.
3. Exactly three records (`baseline`, `candidate`, `orthogonal`) exist for each trial, with the **same** tokenized prefix SHA and configuration. Matched perturbations have equal reported L2 norm at equal alpha.
4. Free-generation trial files are independently rated, completely and while blinded. Trial controls and split provenance are checked before unblinding.
5. A report includes seed count, style effect versus both baseline and orthogonal null, quality tradeoff, sign-flip test, Holm correction, and the pilot-sized p-value limitation.

## Proposed next development

- Cache prefill states and isolate the cost of repeated generation; benchmark VRAM before optimizing.
- Add multiple-rater blinded joins, a response-level semantic equivalence assessor, and a genuinely new technical question family.
- Include a control that injects a direction trained on **neutral vs neutral** shuffled histories. Then replicate with a different model family.
- Define and test a precise state transport law before calling the observed path dependence "holonomy."
