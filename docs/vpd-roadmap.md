# VPD / SAE / holonomy research roadmap

v0.5 includes descriptive open-weight hidden-state capture and a *limited causal experiment* that perturbs whole-block output residuals or rank-one mean-difference directions at the final prompt token, including held-out projection erasure and dose/layer sweeps. It does **not** identify a VPD component or SAE feature. The components and geometric operators below remain a roadmap.

## Open-weight activation capture

- Use an identified open-weight model and tokenizer (for example, a locally runnable 2B–4B chat model) at fixed precision and chat template.
- Capture residual stream activations or layer hidden states at a designated position, with controls for the position's textual and token context.
- Preserve layer, token position, source condition, seed, probe, and checkpoint metadata in typed arrays.
- Compute within-model principal component and sparse-autoencoder activation trajectories; report stability under prompts and decoding settings.

## Causal identification

**Implemented through v0.5:** `phb intervene --mode transplant|rank1` and `phb sweep` for preregistered layers, held-out projection erasure, and norm-matched orthogonal nulls. The rank-one mean-difference candidate is fit on independent seed-varied transcript histories and injected into held-out cases. A norm-matched random direction, neutral-history donor, and identity self-patch provide initial controls. See [intervention protocol](intervention-protocol.md) and [layer sweep protocol](layer-sweep-protocol.md). This answers whether a *tensor-level perturbation* affects next-token continuation scores, not which sparse feature or circuit mediates the behavior.

A candidate "persona component" needs an intervention: suppress or replace the component at selected layers during generation. Compare changes in genre intrusion and technical correctness. Use positive and negative control components to test specificity, repeat across questions, and keep response-style graders blinded.

An association between SAE activations and dramatic writing is not sufficient evidence for a portable persona circuit. Cross-layer causal mediation is a separate measurement.

## Hysteresis and geometric holonomy

A lag between induction and recovery is a **behavioral hysteresis signature**. A genuine geometric holonomy experiment additionally requires:

1. A defined latent state manifold \(\mathcal{M}\) and its coordinates or gauge equivalence classes.
2. A specified connection / state-transport operator between neighboring contexts.
3. A closed contextual loop and a comparison of transported states after returning to the starting point.
4. Reference transformations or controls to separate ordinary token effects, discretization artifacts and coordinate choices from path dependence.
5. Replication across path lengths, directions, and models.

For a candidate connection \(A\) around a path \(\gamma\), the formal object is a path-ordered transport \(U(\gamma)=\mathcal{P}\exp(\oint_\gamma A)\). We must choose and justify \(A\) before numerical values have any holonomy interpretation.

## Percolation interpretation

Define a sparse graph of activated features and interactions after causal validation. Track the size of coherent activation clusters as prompt intensity increases, and test whether a percolation-style threshold predicts abrupt behavioral regime changes. Compare to null graphs preserving degree distributions and feature frequencies. Avoid claiming an actual critical transition on small data.

## Related projects

Potential future adapters: mechanistic decomposition experiments, token SAEs, the user's control harness, and quantized local-model evaluation. Keep proprietary frontier-model behavior comparisons black-box unless model internals are independently accessible.
