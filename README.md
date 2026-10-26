# Persona Holonomy Bench (PHB)

**A reproducible benchmark for transcript-conditioned persona persistence, order sensitivity, and recovery in language models.**

The motivating question is simple: can a long, playful conversation push a language model toward a theatrical response style that persists after the user asks it to return to ordinary technical prose?

The initial seed is the *AI safety monastery / dungeon RPG* conversation. We make this into a controlled protocol rather than trying to decide whether a model has literally reverted to an older model's personality.

> **Scope through v0.12:** transcript-conditioned behavioral hysteresis; forward-only activation capture; rank-one injection and held-out **projection erasure across preregistered decoder layers and strengths**. Paired norm-matched orthogonal controls, first-token diagnostics, unseen technical-probe holdout, exact seed-level sign-flip tests, Holm multiplicity adjustment, and an SVG layer sweep are included. Geometric holonomy and identification of VPD/SAE features remain research hypotheses.


## PercInterp v0.9 replication and provenance gates

The [PercInterp integration](docs/percinterp-bridge.md) now requires the attribution graph's `provenance.model`, optional `provenance.tokenizer`, and **`provenance.training_probe_ids`** to agree with PHB sweep metadata when performing a layer audit. Graph selection remains available on incomplete exploratory graph metadata, but inferential audits fail closed. The audit also reports when different component selectors choose the **same unique decoder blocks**; at block resolution these selectors have indistinguishable causal tests.

The frame-holonomy condition-label null swaps **whole history seeds** across all probes and phases, rather than treating repeated probes or phases as independent permutation units. For scientific interpretation, record the unit count and independent model replication. CI pins a specific PercInterp commit. This release also fixes a PyTorch-version-dependent test assertion in the orthogonal-hook checks. See [v0.9 robustness notes](docs/v09-percinterp-robustness.md).

## v0.12: training-only signed direct influence edges for PercInterp

`phb sae-attribution-plan` checks locally pinned SAE checkpoint hashes, the
frozen Qwen model identity, case provenance, and disjoint seed/question partitions
without loading model weights. `phb sae-attribution` removes **one SAE latent's
ordered-versus-tail-matched contrast contribution** at a time and records the
signed downstream change in the next sampled SAE layer or the first-token style
log-odds. Candidate features and edge weights are obtained from training cases
only. The resulting `phb.percinterp.attribution.v1` graph can be consumed
immediately by `phb perc-select`, followed by `phb sae-feature-map` and the v0.10
held-out feature ablations.

These edges are local finite-difference responses between sampled decoder
blocks, not EAP gradients or an additive explanation of the model. **No pretrained
Qwen weights have yet been evaluated.** See
[the signed attribution protocol](docs/v012-direct-attribution.md) for commands,
resource bounds, provenance, and research limitations.

## Frozen workstation pilot (v0.13)

The six-stage, **offline-first** GPU pilot runner freezes the seed/probe split and artifact hashes before a Qwen model is loaded. It refuses mutable checkpoint-revision labels, missing local HF weights, partial stage outputs, changed manifests, and graph/ablation runs exceeding the declared forward-call cap.

```bash
phb pilot-plan --model /path/to/CACHED_HF_QWEN \
  --revision EXACT_40_HEX_COMMIT_OR_64_HEX_DIGEST \
  --output-dir runs/qwen-001 --repetitions 8 \
  --layers mid,last --top-features 4 --max-model-forwards 250

# Explicit opt-in, one stage at a time, on your model-enabled workstation:
phb pilot-stage --plan runs/qwen-001/plan.json --stage capture
phb pilot-stage --plan runs/qwen-001/plan.json --stage fit
phb pilot-stage --plan runs/qwen-001/plan.json --stage graph
phb pilot-stage --plan runs/qwen-001/plan.json --stage select
phb pilot-stage --plan runs/qwen-001/plan.json --stage map
phb pilot-stage --plan runs/qwen-001/plan.json --stage evaluate
```

See [the workstation pilot protocol](docs/v013-workstation-pilot.md). **No pretrained Qwen effect has been measured yet.** The default eight-seed pilot is intended to test workflow feasibility, not statistical significance.

## Quick start

Python 3.10+; core has **no third-party runtime dependencies**.

```bash
python -m pip install -e '.[test]'
pytest

# Offline end-to-end smoke test. Mock generations are synthetic.
phb run --backend mock --output runs/mock.jsonl
phb analyze --input runs/mock.jsonl --output-prefix runs/mock-report
phb plot --report runs/mock-report.json --output runs/mock-trajectory.svg

# Export exact prompts before spending any inference budget.
phb generate --repetitions 5 --output runs/cases.jsonl
```

## Use an open-weight model via Ollama

With [Ollama](https://ollama.com/) running locally and a chat model already pulled:

```bash
phb run \
  --backend ollama --model qwen2.5:3b \
  --repetitions 5 \
  --phases baseline,peak,exit,washout_1,washout_2 \
  --output runs/qwen.jsonl
phb analyze --input runs/qwen.jsonl --output-prefix runs/qwen-report
```

Change `--model` to your installed model name. Pulling the model is a separate, explicit step. All default inference URLs are loopback addresses.

### Other local runtimes

For vLLM, llama.cpp, or LM Studio exposing a chat-completions API:

```bash
phb run --backend openai-compatible --model YOUR_LOCAL_MODEL \
  --base-url http://127.0.0.1:8000/v1 \
  --repetitions 5 --output runs/compatible.jsonl
```

A hosted compatible endpoint can be specified explicitly. Set its API key in an environment variable, for example `OPENAI_API_KEY`, and review any data-handling implications before transmission. Some backends ignore `seed`; results must be interpreted accordingly.


## Causal rank-one / transplant experiments (v0.4+)

Version 0.4 adds `phb intervene` and `phb intervention-report`. Local HF decoder-block hooks modify the final input-token residual while scoring fixed theatrical and plain continuations. A rank-one direction is trained from the mean difference between two histories and tested on **disjoint, text-varied seed histories**; a norm-matched orthogonal direction and a neutral-history source act as controls.

```bash
pip install -e '.[test,mechanistic]'
phb generate --variation seeded_v1 --repetitions 5 \
  --conditions ordered,tail_matched,neutral --phases exit \
  --probes flaky_ci --output runs/causal-cases.jsonl
phb intervene --model /path/to/local/HF/chat-model \
  --input runs/causal-cases.jsonl --mode rank1 --layer 8 \
  --phases exit --max-pairs 4 --output runs/causal.jsonl
phb intervention-report --input runs/causal.jsonl --output-prefix runs/causal-report
```

This small example verifies pipeline behavior and produces too few independent held-out seeds for confidence intervals. Use more seed-varied histories and probes for scientific conclusions. The rank-one experiment fails closed if training and held-out histories repeat, which is what happened with the **old static seed protocol**. Transplant mode can be used with static histories because it learns no direction from train/test sets.

The causal output is a **difference in mean per-token log-likelihood margins**, not generated reply quality, SAE feature attribution, or measured geometric holonomy. See [the causal protocol](docs/intervention-protocol.md) for equations, controls, and limitations. PyTorch toy-decoder interventions are tested; pretrained open-weight checkpoints have not been run in this environment.


## Held-out causal layer sweep (v0.5)

The next experiment adds `phb sweep`, `phb sweep-report`, and `phb sweep-plot` for **rank-one injection** and **projection erasure** on separate held-out histories. The directional effect is compared to a norm-matched orthogonal intervention and displayed by decoder layer. The readout now distinguishes the first continuation token from later tokens, which matters because patches at the final decoder block can predominantly alter the first-token prediction.

```bash
# Produce content-distinct, seed-matched pairs before any model inference:
phb generate --variation seeded_v1 --repetitions 5 --seed 100 \
  --conditions ordered,tail_matched,neutral --phases exit \
  --probes flaky_ci --output runs/sweep-cases.jsonl

# Cache your chosen model locally; model downloads are opt-in.
pip install -e '.[test,mechanistic]'
phb sweep --model /path/to/local/HF/chat-model \
  --input runs/sweep-cases.jsonl --sources ordered \
  --layers 0,last --alphas 0,1 --phases exit \
  --output runs/sweep.jsonl
phb sweep-report --input runs/sweep.jsonl --output-prefix runs/sweep-report
phb sweep-plot --report runs/sweep-report.json --source ordered \
  --output runs/sweep.svg
```

The five-seed example is a quick mechanistic shakedown. To estimate population-level effects, rerun with at least 15 distinct seeds, multiple probes, and preregistered layers. The first release remains a PyTorch-hook tested implementation; **no pretrained model causal findings are claimed**. See [the full causal layer sweep protocol](docs/layer-sweep-protocol.md) for equations, costs, controls, and interpretation criteria.

## Unseen technical questions and offline cost preflight (v0.6)

The `--heldout-probes` option separates **technical question content** between direction training and evaluation, in addition to holding out independent scripted-history seeds. The report evaluates candidate-versus-orthogonal specificity with seed-clustered two-sided sign-flip tests and Holm-adjusted familywise p-values across all nonzero-dose cells. Very small pilots lack statistical resolution. See the [v0.6 protocol](docs/heldout-inference-protocol.md).

```bash
phb generate --variation seeded_v1 --repetitions 8 --seed 100 \
  --conditions ordered,tail_matched --phases exit \
  --probes flaky_ci,parser_validator \
  --output runs/qwen-cases.jsonl

# No inference, model download, or GPU required:
phb sweep-plan --input runs/qwen-cases.jsonl \
  --layers 0,mid,last --alphas 0,1 --sources ordered \
  --heldout-probes parser_validator --output runs/qwen-plan.json

# Then choose a locally cached, supported chat checkpoint:
phb sweep --model /path/to/local/chat-model \
  --input runs/qwen-cases.jsonl --layers 0,mid,last \
  --alphas 0,1 --sources ordered \
  --heldout-probes parser_validator --device cuda --dtype float16 \
  --output runs/qwen-sweep.jsonl
```

The preflight records the input SHA-256 and deterministic train/test seed assignment, plus an estimated number of model forward calls. It cannot predict peak VRAM or timing. Model weights are never downloaded by default; pretrained causal claims require a real open-weight run. A new experiment should freeze its hypotheses and layers before viewing measured outcomes.

## Held-out free-generation / blinded scoring (v0.7)

The v0.6 causal sweep scores fixed alternative continuations. v0.7 separately
checks **greedy answers the model actually writes**, under baseline, trained
rank-one direction, and equal-norm orthogonal intervention. Only held-out
conversation seeds and held-out technical questions enter evaluation. Exports
hide condition labels and randomize response order for independent human review.

```bash
phb freegen --model /path/to/cached/HF-chat-model \
  --input runs/qwen-cases.jsonl --layer mid --alphas 1 \
  --sources ordered --heldout-probes parser_validator \
  --max-new-tokens 64 --max-total-tokens 1536 \
  --device cuda --dtype float16 --output runs/qwen-freegen.jsonl
phb blind-export --input runs/qwen-freegen.jsonl --output-prefix runs/qwen-blind
# Manually score the shuffled CSV while keeping the PRIVATE key from the rater.
phb blind-report --ratings runs/qwen-blind-ratings.csv \
  --key runs/qwen-blind-PRIVATE-key.json --output runs/qwen-blind-report.json
```

This is a test of actual generated behavior rather than only forced-token
likelihoods. A pretrained checkpoint run and independent blinded ratings remain
outstanding. See [the free-generation validation protocol](docs/freegen-validation.md)
for grading instructions, causal assumptions, and acceptance criteria.



## PercInterp SAE feature-contribution test (v0.10)

The optional `phb perc-components-plan`, `phb perc-components`, and
`phb perc-components-report` commands test a **frozen PercInterp-selected
feature set** at the level of TopK SAE decoder contributions. Candidate ablations
are compared against an equally strong orthogonal perturbation and equal-budget
local-degree/random selectors, with completely held-out history seeds and probe
questions. Only *trusted, SHA-256-pinned local* SAE checkpoints and HF model
weights are loaded. The current test suite exercises synthetic Torch decoder
fixtures; pretrained Qwen results are still needed.

[Read the v0.10 feature-ablation protocol](docs/v010-component-ablation.md)
for the JSON feature-map schema, valid TopK checkpoints, commands and limits.

## PercInterp crossover (v0.8; optional)

PHB now connects to [MoralityLabAI/PercInterp](https://github.com/MoralityLabAI/percolation-interp)
through three CPU-side commands: `phb perc-select` freezes source-to-sink
percolation candidates from externally validated *direct attribution edges*;
`phb perc-audit` compares their explicitly mapped decoder blocks with PHB's
held-out orthogonal-controlled sweep results; and `phb perc-holonomy` constructs
PCA frames from matched PHB activation captures and uses PercInterp's explicit
Procrustes frame connection to measure certified loops and test gauge invariance.

The first two commands **do not** create attribution edges from rank-one
activation differences. The layer audit is a coarse proxy, not component-level
causal validation. The frame experiment is exploratory and requires replication
against a history-shuffle null. Neither needs pretrained weights for graph-only
selection, and all GPU inference remains opt-in.

Install the upstream library with Python 3.11+:

```bash
python -m pip install 'git+https://github.com/MoralityLabAI/percolation-interp'
phb perc-select --graph examples/percinterp/direct-attribution-SYNTHETIC.json \
  --output runs/perc-selection-SYNTHETIC.json
```

See [the complete bridge protocol](docs/percinterp-bridge.md) for actual model
inputs, frozen controls, and both experimental hypotheses.

## Train-only SAE acquisition (v0.11)

PHB can now collect model-local decoder states from **causal-training seeds and
technical probes only**, fit the exact PercInterp `TopKSAE` implementation in a
separate process, and bind the resulting hash-pinned checkpoints to an already
frozen PercInterp selection manifest. The first step requires **no model**:

```bash
phb generate --variation seeded_v1 --repetitions 8 --seed 100 \
  --conditions ordered,tail_matched --phases exit \
  --probes flaky_ci,parser_validator --output runs/sae-cases.jsonl
phb sae-capture-plan --input runs/sae-cases.jsonl --output runs/sae-plan.json
```

With a cached HF checkpoint, run `phb sae-capture`, then `phb sae-fit` to obtain
reconstruction FVU, latent coverage and locally pinned TopK checkpoints.
`phb sae-feature-map` connects those checkpoints to **externally measured direct
attribution** node IDs after `phb perc-select`. PHB does not synthesize graph
edges from activation correlations. See the
[full v0.11 protocol](docs/v011-sae-acquisition.md) for commands, data guards,
and the RTX 5080 pilot budget.

## Design

Each case has an identical system instruction and a held-out **neutral technical question**. The conversation history varies:

| Condition | History before the fixed technical probe |
|---|---|
| `ordered` | Warm-up joke → absurd institutional satire → gothic fantasy scene → exit instruction → up to two neutral washout turns |
| `permuted` | Exactly the same three induction user/assistant pairs in another order, followed by identical exit and washout turns |
| `tail_matched` | Exactly the same final Gothic turn at the same absolute location as `ordered`, with its first two turns swapped |
| `neutral` | Three everyday technical exchanges of similar size, followed by identical exit and washout turns |

We probe the model independently at checkpoints `baseline`, `warmup`, `absurd`, `peak`, `exit`, `washout_1`, `washout_2`. This isolates each probe from subsequent prefixes. The **post-exit** contrast of `ordered` against `permuted` holds the induction-text multiset fixed, testing whether sequence order matters. The `neutral` condition helps diagnose history length and genre exposure; the baseline is a reference with no history.

The assistant messages in the history are fixed scripts. The benchmark therefore tests *response conditioning on transcript content*, and not whether the evaluated model itself would have generated the earlier replies. Add interactive rollouts as a later experiment.

The default run executes 4 conditions × 4 phases × 2 probes × 1 seed = **32 generations**. Inference calls execute in a seeded randomized order to reduce time and backend-drift artifacts; add `--sequential` to disable randomization. Use `--execution-seed` to regenerate the exact call order. Use `--repetitions 5` for a first exploratory set; use more distinct prompts and seeds for publication-quality work.

## Outputs

`run` emits JSONL with `case_id`, condition, phase, seed, full transcript, prompt-size proxy, model/backend details, generation metadata, and output. `analyze` emits JSON and Markdown group summaries, genre-carryover features, seed-clustered paired contrasts, **baseline-to-recovery closure changes**, and **paired difference-in-differences** against the order-matched control. Baseline-to-recovery differences only appear when the run includes `baseline`, and are behavioral measures rather than differential-geometric holonomy. `plot` renders the measured trajectory as a dependency-free SVG. A confidence interval is omitted with fewer than five distinct seeds. The index is a hand-designed lexical heuristic and should be validated against blind human ratings.

**Interpretation boundaries:** Responses that sound like GPT-4.1 are not proof of GPT-4.1 internals in another model. Prompt history can explain apparent persistence in a stateless transformer; calling this "holonomy" demands a defined state space, transport rule, closed paths, and appropriate controls. See [experiment design](docs/experiment-design.md) and [VPD roadmap](docs/vpd-roadmap.md).

## Roadmap

- [x] Order-matched scripted transcripts, recovery checkpoints, seeded probes
- [x] Ollama and OpenAI-compatible local inference adapters
- [x] Transparent style features, paired contrasts, run metadata, CI
- [x] Seeded randomized execution and paired behavioral closure / difference-in-differences
- [ ] Blind human rater protocol and richer independent hold-out probes
- [ ] Interactive model-generated trajectories
- [x] Optional open-weight hidden-state capture (not SAE feature attribution)
- [x] Causal last-prefix-token full-residual transplantation and held-out rank-one mean-difference steering
- [x] Held-out rank-one projection erasure, norm-matched nulls, layer/dose sweep, and first-vs-later token diagnostic
- [ ] Validated SAE feature attribution and layer-specific **SAE** feature ablations
- [ ] Path-transport specification and closed-loop holonomy estimators
- [ ] Percolation-style transition and phase-boundary analysis

## Safety, privacy, reproducibility

All built-in transcripts are fictional and available in the repository. The default backend is an offline mock. The runner does not upload anything by default, and no secrets are written to output. `runs/` is gitignored. Review transcripts, host endpoints and license before testing real unpublished writing, personal messages, or sensitive source material.

MIT licensed. © 2026 Patrick Dugan.

## Publication

The source is published at [patrickdugan/Persona-Holonmy-Bench](https://github.com/patrickdugan/Persona-Holonmy-Bench). The [`examples/`](examples/) directory contains a **synthetic mock-data** report and plot solely to demonstrate output formatting.


## v0.14 Parade ontology: offline simulated-tool pilot (draft PR)

An optional research track studies whether a cheerful fictional ontology changes
the way an assistant handles external-action permissions after the roleplay ends.
This is separate from the frozen enacted-versus-quoted persona preregistration.

The first executable pilot uses four scripted prior-history conditions, two
information styles, two authorization states, two task domains and four history
seeds, yielding **128 deterministic cases**. It supplies a sandboxed approval
service, an attempted-publication log, seed-paired exploratory scoring, and
three explicitly **synthetic** policies to test the scoring infrastructure.

```bash
phb parade-plan --output-dir runs/parade-001 --first-seed 30000 --repetitions 4
phb parade-run --plan runs/parade-001/plan.json --backend mock-policy \
  --output runs/parade-001/policy.jsonl
phb parade-report --plan runs/parade-001/plan.json \
  --input runs/parade-001/policy.jsonl --output runs/parade-001/policy-report.json

# Positive control: a synthetic policy that violates only the enacted+figurative
# draft-only cell. The resulting difference is planted, not discovered in a model.
phb parade-run --plan runs/parade-001/plan.json --backend mock-ontology \
  --output runs/parade-001/ontology.jsonl

# Opt-in actual local model inference, if a compatible model is running.
phb parade-run --plan runs/parade-001/plan.json --backend ollama \
  --model YOUR_LOCALLY_AVAILABLE_MODEL --max-tokens 512 \
  --output runs/parade-001/model.jsonl
```

These tools never call real publishing systems. External model requests are
restricted to loopback HTTP endpoints; every published action is simulated and
permission-checked independently of the model. The initial retrieval response is
represented through a text envelope because some chat APIs lack native tool-role
support. Scripted histories, unequal token positions, and this envelope remain
important scientific limitations. Read
[the full v0.14 design](docs/v014-parade-ontology-holonomy-design.md) and
[the synthetic scenario](examples/parade-ontology-SYNTHETIC.json).
No pretrained model evidence of the Parade Hypothesis exists yet.

For the **geometric square**, export 32 matched conditions to the existing
activation-capture schema, then explicitly opt into local HF model inference:

```bash
phb parade-frame-export --plan runs/parade-001/plan.json \
  --output runs/parade-001/frame-cases.jsonl
phb capture --model /path/to/LOCALLY_CACHED_HF_MODEL \
  --input runs/parade-001/frame-cases.jsonl --layers 14 \
  --output runs/parade-001/frame-activations.jsonl
phb perc-holonomy --activations runs/parade-001/frame-activations.jsonl \
  --conditions neutral,enacted --phases plain,figurative \
  --phase-edge-kind cue_adjacent --layer hs_14 --rank 2 \
  --null-shuffles 15 --output runs/parade-001/frame-report.json
```

The initial four-history-seed capture is for numerical feasibility only.
Seed-cluster permutations, frame overlap, loop eigenphases, and gauge invariance
are handled by the existing PercInterp adapter, with the cue axis correctly
tagged as categorical. An independent, preregistered larger sample and a
second checkpoint are required for scientific generalization.


