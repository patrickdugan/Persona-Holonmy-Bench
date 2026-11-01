# Persona Holonomy Bench: causal layer sweep

Model: `C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554`; reference: `lexical_match`
Readout: `copy_free`
Layers: [9, 18, 27]; strengths: [1.0]
Training seeds: 32; held-out: 32
Train probes: ['flaky_ci', 'parser_validator', 'stale_cache']; evaluation probes: ['feature_flag', 'log_rotation', 'memory_leak', 'rate_limit', 'retry_backoff', 'schema_migration']; probe holdout: True
Holm multiplicity family: 6 nonzero-dose comparisons
Identity max |Δ|: 0.0; alpha-zero max |Δ|: None

Positive **oriented effect** means more Gothic preference after injection or less after erasure.
Specificity subtracts the norm-matched orthogonal null on the same seed/probe.

| Layer | Mode | Donor | Phase | α | Seeds | Pairs | Effect | Null | Specificity | 95% CI | First-token effect | Later-token effect | Sign-flip p | Holm p |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 9 | erase | ordered | exit | 1.00 | 32 | 192 | -0.0279 | +0.0012 | -0.0291 | [-0.0315, -0.0266] | -0.2455 | -0.013911822810769081 | 0.0000 | 0.0002999850007499625 |
| 9 | inject | ordered | exit | 1.00 | 32 | 192 | -0.0429 | -0.0064 | -0.0364 | [-0.0395, -0.0334] | -0.2837 | -0.026912669961651165 | 0.0000 | 0.0002999850007499625 |
| 18 | erase | ordered | exit | 1.00 | 32 | 192 | +0.0714 | +0.0005 | +0.0709 | [0.0678, 0.0737] | +0.1182 | 0.0713238896181186 | 0.0000 | 0.0002999850007499625 |
| 18 | inject | ordered | exit | 1.00 | 32 | 192 | +0.0618 | -0.0043 | +0.0661 | [0.0619, 0.0701] | +0.0468 | 0.0635334129134814 | 0.0000 | 0.0002999850007499625 |
| 27 | erase | ordered | exit | 1.00 | 32 | 192 | -0.0284 | -0.0003 | -0.0281 | [-0.0301, -0.0263] | -0.3542 | -0.0030970368534326553 | 0.0000 | 0.0002999850007499625 |
| 27 | inject | ordered | exit | 1.00 | 32 | 192 | -0.0372 | -0.0008 | -0.0364 | [-0.0386, -0.0342] | -0.5425 | 0.0029073351373275122 | 0.0000 | 0.0002999850007499625 |

## Interpretation limits

- Residual-stream rank-one projection is not SAE-feature ablation or VPD attribution.
- All cases use fixed scripted assistant turns; these are not interactive model rollouts.
- Style readout uses different completion wordings and lengths; prefer external response grading for validation.
- Training and evaluation transcripts differ by seed and share genre and templated syntax; probes may repeat unless held out.
- A single last-prefix-token hook primarily changes the first suffix prediction in final blocks.
- Bootstrap intervals resample transcript seeds; sign-flips operate on seed-averaged paired effects.
- Exact sign-flip p-values assume exchangeability of signs under the null, which must be justified for the design.
- Holm adjustment covers all reported nonzero-dose contrasts and does not rescue post-hoc layer selection.
- Use --heldout-probes for entirely unseen technical readout questions; seeded holdout alone reuses probe text.
- Inspect unregistered layers as exploratory; a best-layer finding requires a fresh validation split.
- No model consciousness, model-identity switch, or geometric holonomy is implied.
