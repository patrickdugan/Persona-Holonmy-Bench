# Causal intervention protocol: v0.4

**Status:** runnable code with PyTorch-hook unit tests. There are no reported experimental findings on pretrained weights yet. Rank-one directions are experimental candidates, not validated VPD components or sparse-autoencoder features.

## Questions and estimands

Can the influence of a scripted fictional conversation on a fixed technical probe be transmitted to another matched transcript through one decoder-block residual stream? If a low-rank direction captures part of this influence, does modifying it *causally* change a pre-registered stylistic readout on held-out conversation histories?

The primary readout is

\[
m(x) = \frac1{|y_{g}|}\sum_{i}\log p(y_{g,i}\mid x,y_{g,<i})
       - \frac1{|y_{p}|}\sum_{j}\log p(y_{p,j}\mid x,y_{p,<j}),
\]

where `g` is a Gothic-styled continuation and `p` is a plain technical continuation. Both answers address the same fixed question; completions are specified in code before model evaluation. **This is a difference of mean per-token log probabilities**, which can be affected by length, tokenization, answer content, and phrasing. It is not a posterior probability that the model “has” a persona.

For a donor source history `s`, target history `t` and selected decoder layer `l`, let `h_l(s)` be the **last prefix token's** decoder-block output (zero-based *block index*). We apply one of two interventions while scoring continuations under `t`:

- **Transplant:** `h_l(t) ← (1−α) h_l(t) + α h_l(s)`. This resamples the complete residual vector.
- **Rank-one steering:** first learn `d_l = mean_train(h_l(s)−h_l(t))` on training seeds, then on disjoint held-out seeds apply `h_l(t) ← h_l(t) + α d_l`. This estimates an *activation-mean-difference candidate*, not an identified feature axis.

The reported effect is `m(patched_t) − m(unpatched_t)`, evaluated on the identical target question and suffixes. Positive values increase the Gothic-versus-plain score. There is an identity self-patch (`h_l(t) ← h_l(t)`) to check numerical stability; a rank-one run also tests an **orthogonal random direction with matched L2 norm**. The `neutral` training condition supplies an additional content-genre contrast.

## Pretrained-model experiment

Install only on a trusted workstation with enough RAM/VRAM for your chosen *text-only causal* chat model. Model loading stays local/cache-only unless `--allow-download` is explicit. `--trust-remote-code` separately opts into executing model-supplied Python.

```bash
pip install -e '.[test,mechanistic]'

# For a short shakedown, vary the actual conversation content across seeds:
phb generate --variation seeded_v1 --repetitions 5 \
  --conditions ordered,tail_matched,neutral \
  --phases exit --probes flaky_ci \
  --output runs/intervention-cases.jsonl

# Use a locally installed/cached Transformers chat model:
phb intervene --input runs/intervention-cases.jsonl \
  --model /path/to/local/model --mode rank1 \
  --layer 8 --max-total-tokens 2048 --max-pairs 4 \
  --output runs/interventions.jsonl

phb intervention-report --input runs/interventions.jsonl \
  --output-prefix runs/intervention-report
```

To examine a full-activation transplant rather than a learned direction:

```bash
phb intervene --input runs/intervention-cases.jsonl \
  --model /path/to/local/model --mode transplant --layer 8 \
  --output runs/transplant.jsonl
```

For a preliminary report, preregister at least five held-out *unique transcript seeds* (for example, 15 total seeds with `--holdout-fraction .4`) plus all three probes and both exit and washout phases. A single short shakedown yields no confidence interval. Avoid selecting the best layer from the held-out runs; preregister layers or reserve another validation partition first. Default phase selection targets `exit,washout_2`; the input case file must contain both if defaults are used.

## Controls, pitfalls, and failure modes

1. **History leakage:** The v0.3 static protocol changed seed labels without changing `ordered`/`tail_matched` conversation text. For rank-one experiments, v0.4 checks SHA-256 hashes of the full scripted history and *rejects* collisions across training and test seeds. Use `phb generate --variation seeded_v1`. Within this protocol, neighboring seeds have unique text patterns for the first 256 seeds by construction; the guard remains active for arbitrary seed ranges. All conditions for a given seed keep the same induction multiset, retaining the order-matched comparison.
2. **System and probe equality:** The full prefix is replayed per case, matched cases use the same held-out probe, and the source/target need the same system instruction. A role-ending mismatch or missing pair fails closed.
3. **No padding or truncation:** The probe scores a single conversation at a time. The total token budget includes the full prefix and each continuation; over-limit cases raise an error.
4. **No multi-turn hidden state:** This is an autoregressive model with a long prompt, so the contextual activation is reconstructible from tokens. The word *persona* names an operational style profile only.
5. **Causal scope:** A single boundary-token residual patch changes output likelihood. A large effect can come from many correlated functions encoded in one residual vector. Neither transplant nor mean steering demonstrates feature-level necessity/sufficiency. A targeted SAE feature ablation and cross-prompt replication would be additional evidence.
6. **Quality:** The Gothic continuation deliberately contains different words and punctuation. Validate the measured effect with blind ratings of generated responses and matched semantic correctness; the logprob statistic itself is a surrogate.
7. **Architecture:** Tested with a small PyTorch two-layer causal decoder fixture. The adapter supports common Hugging Face `model.layers`, `transformer.h`, `gpt_neox.layers`, and `model.decoder.layers` stacks. Architecture-specific fused layers, hooks that run more than once, and some remote-code wrappers may require adaptation. Full pretrained-model runs have not yet been verified in this repository.
8. **Numerics:** Inspect `self_patch_delta` and `self_patch_check_passed_1e_3`. Identity patch errors exceeding `1e-3` in the mean-token margin invalidate the run pending debugging. Model precision, quantization and tokenizers may change the magnitude of effects.
9. **Statistics:** The report averages probe effects **within each seed** before its bootstrap; at least five independent held-out seeds are required for an interval. Repeated probes sharing an identical conversation induction are *not* independent seeds.

## What would falsify the hypothesis?

A rank-one direction fails to transfer if its held-out mean margin change does not separate from a norm-matched orthogonal random direction or the neutral-history direction across varied probes. A transplant can show a causal change even when rank-one transfer fails: that would argue against a single stable persona axis at the selected block. Success in these tests would justify a more expensive SAE and ablation study; it would still not establish geometric holonomy, which needs a transport operator.
