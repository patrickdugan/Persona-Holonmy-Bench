# Persona Holonomy Bench: run report

Recorded generations: **210**

**SYNTHETIC MOCK DATA. No empirical model finding is implied.**

## Behavior by condition and phase

| Condition | Phase | N | Persona index | Genre intrusion | History chars |
|---|---|---:|---:|---:|---:|
| neutral | absurd | 10 | 0.000 | 0.000 | 526 |
| neutral | baseline | 10 | 0.000 | 0.000 | 0 |
| neutral | exit | 10 | 0.000 | 0.000 | 982 |
| neutral | peak | 10 | 0.000 | 0.000 | 776 |
| neutral | warmup | 10 | 0.000 | 0.000 | 267 |
| neutral | washout_1 | 10 | 0.000 | 0.000 | 1179 |
| neutral | washout_2 | 10 | 0.000 | 0.000 | 1359 |
| ordered | absurd | 10 | 0.000 | 0.000 | 527 |
| ordered | baseline | 10 | 0.000 | 0.000 | 0 |
| ordered | exit | 10 | 0.950 | 1.000 | 1016 |
| ordered | peak | 10 | 0.950 | 1.000 | 810 |
| ordered | warmup | 10 | 0.000 | 0.000 | 269 |
| ordered | washout_1 | 10 | 0.950 | 1.000 | 1213 |
| ordered | washout_2 | 10 | 0.000 | 0.000 | 1393 |
| permuted | absurd | 10 | 0.760 | 0.800 | 545 |
| permuted | baseline | 10 | 0.000 | 0.000 | 0 |
| permuted | exit | 10 | 0.950 | 1.000 | 1016 |
| permuted | peak | 10 | 0.950 | 1.000 | 810 |
| permuted | warmup | 10 | 0.380 | 0.400 | 272 |
| permuted | washout_1 | 10 | 0.950 | 1.000 | 1213 |
| permuted | washout_2 | 10 | 0.000 | 0.000 | 1393 |

## Paired comparisons

Positive means the ordered trajectory had a larger persona index.

| Phase | Contrast | Paired probes | Seeds | Δ mean | Seed-bootstrap 95% CI |
|---|---|---:|---:|---:|---|
| peak | ordered minus permuted | 10 | 5 | +0.000 | [0.000, 0.000] |
| peak | ordered minus neutral | 10 | 5 | +0.950 | [0.950, 0.950] |
| exit | ordered minus permuted | 10 | 5 | +0.000 | [0.000, 0.000] |
| exit | ordered minus neutral | 10 | 5 | +0.950 | [0.950, 0.950] |
| washout_1 | ordered minus permuted | 10 | 5 | +0.000 | [0.000, 0.000] |
| washout_1 | ordered minus neutral | 10 | 5 | +0.950 | [0.950, 0.950] |
| washout_2 | ordered minus permuted | 10 | 5 | +0.000 | [0.000, 0.000] |
| washout_2 | ordered minus neutral | 10 | 5 | +0.000 | [0.000, 0.000] |

## Interpretation limits

- Scripted assistant turns: tests transcript conditioning, not autonomous multi-turn rollouts.
- Lexical features are exploratory proxies requiring blind human validation.
- Bootstrap intervals resample seeds; fewer than five seeds yield no interval.
- The setup cannot directly identify GPT-4.1-like internals or geometric holonomy.
- Context length, tokenization, and recency remain possible confounders.
