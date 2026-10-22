# Causal layer sweep protocol (v0.5)

**Status:** implemented and tested on a small PyTorch causal decoder. No pretrained weights have been evaluated in this development environment. This procedure targets an **activation-mean-difference candidate**, neither a VPD factor nor a trained SAE feature.

## Question

At which decoder blocks, if any, does a conversation-derived rank-one direction cause a reproducible change in the model's preference for theatrical over ordinary technical continuations? Is the same direction required for that style preference on the donor transcript? Does either result exceed a matched random perturbation?

The default donor is `ordered`, the reference/recipient control is `tail_matched`, and the analysis starts *after the explicit exit instruction*. Both transcripts contain the same induction turns and share the final theatrical induction turn. Only the order of the earlier turns changes. `neutral` is a useful separate genre contrast with different content; interpret that comparison separately.

## Equations and identification

For decoder block `ℓ`, define `h_ℓ(x)` as the output residual vector at the last **prefix** token of transcript `x`. The mean-difference candidate and reference center are estimated on training seeds only:

\[
 d_\ell = \frac{1}{N_{\rm train}}\sum_{i\in\mathrm{train}} [h_\ell(s_i)-h_\ell(t_i)],\qquad
 \mu_{t,\ell}=\frac{1}{N_{\rm train}}\sum_{i\in\mathrm{train}}h_\ell(t_i), \qquad
 u_\ell=d_\ell/\lVert d_\ell\rVert_2.
\]

Each pair `(s_i, t_i)` has matching seed, phase, final technical probe, and system instruction. The split is **by independent history seed**; any train/test duplicated transcript history triggers an error.

**Injection** into a held-out reference `t` tests a limited form of causal sufficiency:

\[
 h'_\ell(t) = h_\ell(t) + \alpha d_\ell.
\]

**Projection erasure** on a held-out donor `s` tests whether that particular residual direction contributes causally to the readout:

\[
 h'_\ell(s) = h_\ell(s) - \alpha\langle h_\ell(s)-\mu_{t,\ell}, u_\ell\rangle u_\ell.
\]

These are interventions on one position at one decoder block. The erasure removes a one-dimensional projection relative to the training-reference mean; it cannot establish that the feature is essential to the model's overall ability to write Gothic prose.

**Null interventions** use a deterministically generated random unit direction orthogonal to `d`. Injection nulls have the same norm as `α d`; erasure nulls have the same norm as the *actual removed projection on that held-out sample*. `α=0` serves as a sham. A self-transplant verifies that the hook leaves activations unchanged when it copies the same vector back.

The outcome is the shift in the difference of *mean per-token log-likelihood* of two preselected continuations (`theatrical` minus `plain`), under identical prefix/probe. The direction of the reported effect is oriented so that a positive value means **more theatrical on injection** or **less theatrical on erasure**. We subtract the same recipient's norm-matched null outcome to report **paired specificity**.

Every intervention separately records changes to the **first** suffix token's log-probability margin and to the mean margin of **later** tokens. At the final decoder block, the hooked prefix output directly affects the first continuation-token prediction, while later continuation-token activations at that block are already computed. A final-block signal concentrated on the first token is therefore an expected architectural artifact, not strong evidence for sustained persona behavior.

## A cheap local shakedown

The following uses a previously cached or locally downloaded Transformers text-only causal chat model and keeps **model downloads disabled by default**. On a 16 GB RTX 5080, begin with an actual model that fits its memory budget at the selected precision.

```bash
python -m pip install -e '.[test,mechanistic]'

phb generate --variation seeded_v1 --repetitions 5 --seed 100 \
  --conditions ordered,tail_matched,neutral --phases exit \
  --probes flaky_ci --output runs/cases-shakedown.jsonl

phb sweep --model /path/to/local/HF-chat-model \
  --input runs/cases-shakedown.jsonl \
  --sources ordered --reference tail_matched --phases exit \
  --layers 0,last --alphas 0,1 \
  --max-total-tokens 1536 --device cuda --dtype float16 \
  --output runs/sweep-shakedown.jsonl

phb sweep-report --input runs/sweep-shakedown.jsonl \
  --output-prefix runs/sweep-shakedown
phb sweep-plot --report runs/sweep-shakedown.json \
  --source ordered --phase exit --alpha 1 \
  --output runs/sweep-shakedown.svg
```

This has only two held-out transcript seeds. It tests plumbing, not a scientific claim. The default reports omit 95% intervals until **five independent held-out seeds** exist.

## Pre-registered main run

Use at least 15 distinct seeded transcripts, with all three technical probes, `exit` and optionally `washout_2`, to improve generalization. Choose the layers and strengths **before** examining held-out scores, such as `4,mid,last` and `0,0.5,1`. A 15-seed, 3-probe, 2-source, 3-layer sweep involves hundreds of paired forced-scoring calls; expect substantial inference cost. Use `--sources ordered` and `--phases exit` for a first full run, and run the neutral-history contrast separately. A held-out seed contains multiple probes, so confidence intervals cluster by seed. The same probe templates remain in training; future leave-one-probe-family-out validation is necessary to support broader generalization.

```bash
phb generate --variation seeded_v1 --repetitions 15 --seed 100 \
  --conditions ordered,tail_matched,neutral --phases exit,washout_2 \
  --probes flaky_ci,parser_validator,stale_cache \
  --output runs/main-cases.jsonl

phb sweep --model /path/to/local/HF-chat-model \
  --input runs/main-cases.jsonl --sources ordered \
  --phases exit,washout_2 --layers 4,mid,last --alphas 0,0.5,1 \
  --output runs/main-sweep.jsonl

phb sweep-report --input runs/main-sweep.jsonl \
  --output-prefix runs/main-sweep-report
phb sweep-plot --report runs/main-sweep-report.json --source ordered \
  --phase exit --alpha 1 --output runs/main-sweep.svg
```

Record exact model revision, local weights hash, tokenizer version, precision, GPU and PyTorch/Transformers versions alongside exported cases before drawing conclusions. Source JSONL exposes full scripted prompts; the runner avoids implicit model downloads and remote APIs.

## Stopping rules

- **Valid causal direction evidence:** self-transplant identity check passes, alpha-zero effect equals zero, held-out directional specificity differs from its paired null reproducibly, the sign generalizes across probes and seeds, and the effect appears beyond the first forced token in an earlier block. Interpret effect sizes using controls and generated-response ratings.
- **Weak evidence:** large mean shift with near-equal null shift, effect present only in a final-block first token, or effect only on one registered wording. Such results motivate revised readouts and fresh evaluation data.
- **Evidence against a stable rank-one persona direction:** injection and erasure produce noisy or null-matched results across registered blocks and seeds, even when whole-residual transplantation succeeds. The behavior may be diffuse, nonlinear, or readily explained by surface prompt content.
- **Holonomy:** never infer geometric holonomy from these data alone. A connection and state-transport operator on a defined representation manifold are required before a closed-loop calculation can carry that name.

The SVG plots summarize exploratory layer effects; a layer selected after inspecting the holdout needs **another independent validation partition** before significance claims.
