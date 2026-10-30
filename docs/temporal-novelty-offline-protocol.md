# Frozen offline temporal prediction versus novelty protocol (v0.1)

Status: **design frozen before any authorized activation-data analysis**. Synthetic code validation only. This document is not the repository's frozen confirmatory persona study.

## Scope and grounding
Primary: Lubana et al., *Priors in Time: Missing Inductive Biases for Language Model Interpretability*, ICLR 2026, https://arxiv.org/abs/2511.01836 .
Source index: https://github.com/patrickdugan/Goodfireverse .
This is a standard-library diagonal AR(1) predictor and residual diagnostic. It neither trains an SAE nor recreates Temporal SAE/Temporal Feature Analysis. The residual measures one-step forecast error, and does not identify semantic novelty by itself.

## Registered analysis
1. Unit = independent whole history identified by unique seed; repeated probes/time steps are correlated observations, never extra units. Use a predeclared disjoint, deterministic split by history/seed. Do not choose the split based on measured outputs.
2. Fit diagonal linear one-step prediction `x[t,j] = intercept[j] + slope[j]*x[t-1,j]` with positive ridge regularization, using transitions from **training** histories only. Prediction at held-out time t uses observed x[t-1] and frozen coefficients; no future state, future labels, or test history ever enters fit.
3. Comparator A is each dimension's training target mean. Comparator B refits the same predictor after independently shuffling **training** history order with recorded seed 731. Score controls on the identical untouched held-out histories.
4. Primary descriptive endpoints: per-history average one-step MSE, static MSE, and shuffle MSE. Aggregate per-history metrics with equal unit weight. Secondary: mean residual norm at predeclared event indices versus other indices, with explicit event-count denominators.
5. Deterministic synthetic families: stationary, drifting, order-sensitive, and planted event-boundary. Additionally verify exact linear predictability, causal prefix invariance, deterministic replay, held-out-fit rejection and duplicated-seed rejection.
6. Freeze all preprocessing, prompt/task family, model hash, layer, token positions, activation extraction method, history/seed partitions, event annotation rules, exclusions, ridge, shuffle seed, metrics and multiplicity plan **before** external analysis. No hyperparameter selection on held-out data.
7. For future empirical work, obtain explicit authority and a supplied local activation archive. Validate finite numeric fixed-dimensional vectors, unique independent history IDs, provenance, explicit authorization, dataset hash, and split before ingest. This version has **no activation archive loader** and cannot initiate a model call.
8. Negative outcomes remain reportable: linear cases with no added value, lack of event localization, shuffled predictor matching the ordered fit, and baseline superiority.

## Reporting and limitations
For each evaluation show exact code SHA, input hash, group counts, steps per group, per-group scores, event denominators, excluded units and predefined metrics. Any inferential uncertainty needs group-level bootstrap/permutation across independent histories, with multiplicity stated. Do not infer persona states, model holonomy, causal intervention effects, semantic novelty, or paper reproduction from synthetic fixtures. The toy AR(1) misses nonlinear or long-context predictability; autoregressive observation of a prior held-out step is permissible for one-step forecasting and must not be relabeled a zero-shot rollout.
Run locally without services:
```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_temporal_novelty_offline.py'
PYTHONPATH=src python -m persona_holonomy.temporal_novelty_offline --kind event_boundary
```
