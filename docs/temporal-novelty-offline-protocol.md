# Frozen offline temporal prediction versus novelty protocol (v0.2)

**Status:** Exploratory diagnostic design fixed before any authorized activation-data analysis. This synthetic-only work is separate from the existing PHB confirmatory study, Parade, PercInterp, and manifold-oracle tracks. The protocol freeze applies to this version and must be explicitly revised before a new empirical study.

Primary grounding: Lubana et al., *Priors in Time: Missing Inductive Biases for Language Model Interpretability*, ICLR 2026, https://arxiv.org/abs/2511.01836 and https://openreview.net/forum?id=4J2e3nWiC8 . Source index: https://github.com/patrickdugan/Goodfireverse . This is a standard-library diagonal AR(1) predictor/residual diagnostic, not the paper's Temporal SAE architecture or Temporal Feature Analysis.

## Frozen experimental method

1. The independent unit is an entire trajectory seed. Split by **whole seed** before any fitting, preprocessing or evaluation. Repeat probes may share a seed but require distinct `probe_id`; they remain one independent evaluation unit. Reject the same seed across training and test or duplicated seed/probe identities. Default synthetic split uses seeds 0–7 for train and 8–15 for test. No future-state features or held-out labels enter fit.
2. Fit the diagonal AR(1) `x[t,j] = intercept[j] + slope[j] × x[t-1,j]` using *training-only* adjacent state pairs, ridge `1e-9`. Weight each training seed equally across its probes and time steps. The step-level residual `x[t] - predicted[t]` describes unexpected numerical activation changes in a one-step teacher-forced setting. It does not establish semantic novelty.
3. Freeze and evaluate three controls on the same held-out histories: **static mean difference** `x[t-1] + (mean_train(x[t]) - mean_train(x[t-1]))`, a constant training target mean predictor, and a diagonal AR(1) fit on independently shuffled **training-only** states (fixed shuffle seed `731`). The mean-difference comparator is a mean *step* difference, not a condition-versus-condition persona direction. To test an actual condition contrast, register paired condition labels and a separate contrast estimator before collecting activations.
4. The exact training input is bound to each fitted model with a SHA-256 fingerprint of sorted numeric histories, independent of control shuffling. Reject a shuffled comparator fitted to any other source, even with identical seed labels. Primary endpoint: group-mean one-step squared error, averaged first within each probe then equally across probes of one seed, then equally across independent test seeds. Compare paired differences `(control MSE - temporal MSE)` at the seed level. Positive means an AR(1) MSE advantage. Report all outcomes even where the temporal method adds nothing.
5. Descriptive paired uncertainty: fixed-seed bootstrap (`20261031`, 2,000 resamples), resampling independent seeds with replacement; percentile 95% interval. Fewer than eight held-out seeds yields `insufficient_independent_units` with no interval. Intervals on synthetic controls are software diagnostics, not general evidence about model behavior or a calibrated confirmatory significance test.
6. Secondary event-boundary endpoint: event residual L2 norm versus non-event residual L2 norm on held-out trajectories. Event locations must be fixed without inspecting evaluated residuals. Record event counts. The deterministic mock puts a jump at step `length//2` **only on held-out event-boundary seeds**, so training never sees the planted shock.
7. Families: stationary autoregression, drifting process, sign-reversing/order-sensitive autoregression, and planted event boundary. Also test exact linear dynamics, matching-endpoint limits where possible, exact deterministic replay, leakage guards, prefix causality, duplicate rejection, repeated-probe grouping, and a simple drift where the static mean-step predictor is sufficient.
8. For any future empirical extension, first acquire **explicit permission and supplied saved activation data**. This version has **no archive loader** and performs no model calls. Freeze the model/checkpoint revision and tokenizer, layer, token-position rule, complete input hash, unique seed/probe IDs, event annotation, training/test seed partitions, preprocessing, ridge, controls, outcome priority, and exclusions before analysis. Save immutable versions of outputs.

## Required report

Record exact source SHA and input digests; training/test seed lists; counts of independent units, probes, steps and events; per-seed and per-probe metrics; all control MSEs; paired improvement and intervals; all excluded/missing histories; and any protocol deviation. Repeated steps are correlated, not additional independent observations. Bootstrap intervals are descriptive and do not correct for testing multiple families/endpoints. Report null or reversed performance and insufficient-unit outcomes.

**Limitations:** A diagonal AR(1) cannot learn nonlinear or long-context predictability. The observed previous held-out state at time `t-1` is allowed for one-step evaluation and must not be called an autonomous multi-step forecast. Synthetic event detection is not evidence for real semantic novelty, persona persistence, causal representation, hidden holonomy, or reproduction of the ICLR study. No inference about real models is available without authorized activations and an independently registered evaluation.

Run the isolated diagnostic without external services:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_temporal_novelty_offline.py' -v
PYTHONPATH=src python -m persona_holonomy.temporal_novelty_offline --kind event_boundary
```
