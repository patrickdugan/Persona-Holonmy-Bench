# PHB v0.9: PercInterp null and provenance hardening

**This release is a reproducibility and validity patch.** It does not report effects from pretrained Qwen or identify VPD factors.

## A. Verifiable layer-audit preconditions

The attribution graph must export at least:

```json
{
  "provenance": {
    "edge_semantics": "direct_attribution",
    "estimator": "PREDECLARED_ESTIMATOR",
    "model": "EXACT_MODEL_ID_USED_BY_SWEEP",
    "tokenizer": "OPTIONAL_EXACT_TOKENIZER_ID",
    "train_seeds": [100, 101, 103],
    "training_probe_ids": ["flaky_ci"]
  }
}
```

For \`perc-audit\`, \`model\`, \`train_seeds\`, and \`training_probe_ids\` must match the sweep. If provided, \`tokenizer\` must also match. A JSON label alone is not an attestation of model-weight identity. Record the immutable checkpoint SHA and tokenizer revision externally; populate both graph and sweep with stable identifiers.

The adapter rejects missing train/test probe partitions and tests the same held-out seed/probe cells across all selected decoder blocks. Selector identity must be assessed at two resolutions: graph **components** and PHB **decoder blocks**. Two different component sets may collapse to the same block set. These block-level selectors are experimentally identical under PHB's current residual interventions; the audit explicitly labels such cases, and cannot claim percolation has found better components.

## B. Seed-clustered label null for geometric frames

An independent experiment seed can contain several technical probes measured at multiple phases. The previous implementation shuffled the labels of individual observations, treating highly correlated items as independently exchangeable. The v0.9 null samples one condition swap bit **per seed**, and uses it for every probe and phase in that seed. The null therefore preserves within-seed trajectories and correlations.

The per-draw seed-swap mask is recorded as a SHA-256 fingerprint for reproducibility. Duplicating strongly correlated probe rows can change numerical PCA outputs even with an identical assignment, especially near singular-value degeneracies; test invariance of the randomization unit and masks rather than exact loop statistics. The reported statistic is still the maximum certified loop Frobenius distance after PCA fitting and polar frame transport. Local-basis gauge invariance remains a regression check. The Monte Carlo p-value is exploratory, and its resolution depends on permutation draws plus the number of independently exchangeable seeds. The connection is specified by analysis, not learned as the model's own dynamic transport.

## C. Testing and dependencies

The offline core and PyTorch hook tests run without PercInterp. A dedicated CI job installs **MoralityLabAI/percolation-interp commit \`c0b3f6abf8e847292aaf1e21b8f4b804fd4d8ce0\`** and tests graph and holonomy adapters against the actual published API. New tests reject swapped model/probe provenance, identify identical block selectors, and verify that doubling probe rows within existing seeds does not inflate the number of permutation units.

## D. Gate for advancing the scientific claim

1. Run a pretrained HF Qwen checkpoint with pinned model, tokenizer, and captured sweep hashes.
2. Derive graph edges from **direct attribution** on the same training subset. PHB sweep log-probability margins and cosine distances are not valid edge substitutes.
3. Freeze the percolation selection before evaluation. Compare selectors using matched budgets on held-out samples; then implement component-level interventions before claiming per-component causal effects.
4. Replicate activation-frame holonomy with independent seeds and at least one additional checkpoint. Report weak-overlap link failures and null results.

**Outcome to date:** CI and local synthetic tests establish plumbing and methodological checks. No Qwen pretrained causal or holonomy effects have been measured.
