# PHB v0.6: unseen-probe transfer, randomization inference, and cost preflight

## Registered question

Can the residual-space direction fitted to **training-seed conversation histories and training technical questions** selectively shift the measured style margin on **new conversation seeds and an unseen technical question**?

This is evidence about a chosen causal intervention and a chosen log-likelihood readout. It does **not** identify an SAE feature, show an older model is present, or establish geometric holonomy.

## Primary test and estimand

Pre-register the source (`ordered`), reference (`tail_matched`), model revision, seed range, readout prompts, layer list, doses, and test probe **before observing results**. `tail_matched` changes the order of earlier scripted user/assistant pairs but keeps the final fictional induction turn in the same position. The script retains the same system prompt, user question, exit instruction, and answer alternatives.

On training pairs, estimate decoder-block direction

\[d_\ell = \frac{1}{N_{tr}} \sum_{i \in tr} (h_{i,ordered,\ell} - h_{i,tail,\ell}).\]

For held-out source/recipient pair *i*, record the log-likelihood margin difference (theatrical minus ordinary answer). The paired *specificity* is

\[z_i = (m^{candidate}_{i} - m^{baseline}_i) - (m^{orthogonal}_{i} - m^{baseline}_i),\]

with the sign reversed for projection erasure so positive always means the pre-registered direction behaved as hypothesized. Average multiple probes within each **seed** before statistical inference; treat seeds, rather than tokens or continuations, as the independent unit.

Both candidate and orthogonal controls have matched perturbation norm for each case and dose. Norm matching cannot exclude every distributional or semantic confound.

## Split and leakage guard

`--variation seeded_v1` changes the actual wording and fantasy imagery for each seed. The seed split is deterministic and reused by `phb sweep-plan` and `phb sweep`.

`--heldout-probes parser_validator` learns only from the remaining technical probes and evaluates on `parser_validator` for held-out seeds. Every selected `--heldout-probes` value must already be present in the exported JSONL. There must be at least one training probe. Repeated transcript histories across the seed split raise an error, even if the case IDs differ.

Question holdout is **text holdout**, not guaranteed semantic OOD transfer: all technical probes still come from a small, preregistered family. For stronger generalization, author new question domains and freeze them before testing.

## Inference

For each pre-registered nonzero-dose cell, PHB reports a two-sided sign-flip test on the **seed-averaged candidate-minus-orthogonal specificity**. With at most 16 independent test seeds it enumerates all sign combinations (exact conditional sign-flip test); above 16 it uses 20,000 deterministic Monte Carlo flips with a plus-one correction. The test's exchangeable-sign assumption requires judgment about the design.

The first report includes all nonzero-dose layer × intervention × source × phase combinations in one Holm familywise multiplicity correction. Alpha zero is an identity check and receives no Holm value. This conservative family is defined by the CLI inputs, and it does not validate a layer selected after inspecting data. Treat alpha/dose curves and unregistered layer selection as exploratory until reproduced on a fresh test split.

For n independent seeds, an all-same-sign two-sided exact test has minimum p = 2 / 2^n. With only 2 test seeds, p cannot be below 0.5; with 3, it cannot be below 0.25. Thus the 5- or 8-seed runs below are **mechanistic shakedowns**, and apparent effect sizes deserve caution.

## Preflight on a small local checkpoint

```bash
pip install -e '.[test,mechanistic]'
phb generate --variation seeded_v1 --repetitions 8 --seed 100 \
  --conditions ordered,tail_matched --phases exit \
  --probes flaky_ci,parser_validator \
  --output runs/pilot-cases.jsonl

phb sweep-plan --input runs/pilot-cases.jsonl \
  --sources ordered --layers 0,mid,last --alphas 0,1 \
  --heldout-probes parser_validator \
  --output runs/pilot-plan.json

phb sweep --model /path/to/local/chat-checkpoint \
  --input runs/pilot-cases.jsonl --sources ordered \
  --layers 0,mid,last --alphas 0,1 \
  --heldout-probes parser_validator --phases exit \
  --device cuda --dtype float16 \
  --output runs/pilot-sweep.jsonl
phb sweep-report --input runs/pilot-sweep.jsonl \
  --output-prefix runs/pilot-report
phb sweep-plot --report runs/pilot-report.json \
  --source ordered --phase exit --alpha 1 \
  --output runs/pilot-plot.svg
```

`phb sweep-plan` emits the input SHA-256, split, number of paired cases, requested layer names, and expected number of model forward invocations. **It cannot estimate tokenized sequence length, GPU memory, speed, or FLOPs without the actual model and tokenizer.** Self-patch identity checks are enabled by default and cost additional forwards. The planner accounts for their calls.

No weights are downloaded without `--allow-download`; `--trust-remote-code` is opt-in. The checkpoint should be cached and have a supported decoder stack plus a working chat template. A model-side architecture or tokenizer incompatibility must be resolved before calling results scientific.

## Full experiment and reporting

Increase to at least 30 seed-distinct histories, preferably expanding the set of pre-registered probes. Fix the split before viewing any intervention outputs, record exact model commit/revision, software environment, generation templates, GPU, dtype, and resource usage. Replicate with a new independent seed range and, ideally, another model family.

The fixed continuations differ in wording and token length, and log likelihood of a written continuation is not the same quantity as the probability of generating that style freely. Independently validate the direction with blinded ratings of actual model responses and neutral questions unrelated to this prompt family. Report null and negative results.

For geometric holonomy, additionally define a state representation and a transport/connection operation and compare closed-loop transport with appropriate controls. Residual-space differences caused by rearranging a text transcript are **path dependence**, not mathematical holonomy by themselves.
