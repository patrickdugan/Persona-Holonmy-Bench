# Persona Holonomy Bench (PHB)

**A reproducible benchmark for transcript-conditioned persona persistence, order sensitivity, and recovery in language models.**

The motivating question is simple: can a long, playful conversation push a language model toward a theatrical response style that persists after the user asks it to return to ordinary technical prose?

The initial seed is the *AI safety monastery / dungeon RPG* conversation. We make this into a controlled protocol rather than trying to decide whether a model has literally reverted to an older model's personality.

> **Scope of v0.2:** black-box behavioral probes of scripted conversation histories, an order-matched control, a neutral-history control, recovery checkpoints, local inference adapters, and paired descriptive analysis. No internal-activation measurement is claimed. Geometric holonomy is a research hypothesis requiring a precise transport operator and causal validation.

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

## Design

Each case has an identical system instruction and a held-out **neutral technical question**. The conversation history varies:

| Condition | History before the fixed technical probe |
|---|---|
| `ordered` | Warm-up joke → absurd institutional satire → gothic fantasy scene → exit instruction → up to two neutral washout turns |
| `permuted` | Exactly the same three induction user/assistant pairs in another order, followed by identical exit and washout turns |
| `neutral` | Three everyday technical exchanges of similar size, followed by identical exit and washout turns |

We probe the model independently at checkpoints `baseline`, `warmup`, `absurd`, `peak`, `exit`, `washout_1`, `washout_2`. This isolates each probe from subsequent prefixes. The **post-exit** contrast of `ordered` against `permuted` holds the induction-text multiset fixed, testing whether sequence order matters. The `neutral` condition helps diagnose history length and genre exposure; the baseline is a reference with no history.

The assistant messages in the history are fixed scripts. The benchmark therefore tests *response conditioning on transcript content*, and not whether the evaluated model itself would have generated the earlier replies. Add interactive rollouts as a later experiment.

The default run executes 3 conditions × 4 phases × 2 probes × 1 seed = **24 generations**. Inference calls execute in a seeded randomized order to reduce time and backend-drift artifacts; add `--sequential` to disable randomization. Use `--execution-seed` to regenerate the exact call order. Use `--repetitions 5` for a first exploratory set; use more distinct prompts and seeds for publication-quality work.

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
- [ ] Open-weight residual stream and SAE activation capture
- [ ] Causal component interventions, activation patching, and layer-specific ablations
- [ ] Path-transport specification and closed-loop holonomy estimators
- [ ] Percolation-style transition and phase-boundary analysis

## Safety, privacy, reproducibility

All built-in transcripts are fictional and available in the repository. The default backend is an offline mock. The runner does not upload anything by default, and no secrets are written to output. `runs/` is gitignored. Review transcripts, host endpoints and license before testing real unpublished writing, personal messages, or sensitive source material.

MIT licensed. © 2026 Patrick Dugan.

## Publication

An extracted copy can be published with GitHub CLI using the commands in [PUBLISHING.md](PUBLISHING.md). The [`examples/`](examples/) directory contains a **synthetic mock-data** report and plot solely to demonstrate output formatting.
