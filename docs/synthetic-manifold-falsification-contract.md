# Synthetic manifold transport falsification: future evaluation contract (draft v0.1)

Status: **preregistration-ready proposal, not preregistered and not empirical evidence**.
Scope: isolated CPU-only oracle and tests. No models, checkpoints, neural activations, training, inference, external calls, or modifications to frozen PHB cases. Temporal diagnostics remain independently owned.

## Mathematical oracle and falsifiable null

Let a polygonal path have vertices x_i=(u_i,v_i), with toy connection A_k=(-kv/2,ku/2).
Its exact segment integral is T(P;a,k)=(ak/2) Σ_i(u_i v_{i+1}-v_i u_{i+1}).
The endpoint-only comparator is L(P;a,w)=a w·(x_end-x_start).
A planar SO(2) basis change leaves T invariant when the connection is transformed consistently. Reflection, arbitrary rescaling, nonlinear reparameterization without a corresponding connection transform, and general neural representation gauge changes are **not** covered by this invariance check.

Predeclared synthetic checks:
1. **Zero dose:** T(P;0,k)=0 and L(P;0,w)=0.
2. **Reverse:** T(reverse(P);a,k)=-T(P;a,k), including a closed loop.
3. **Matched endpoints:** compare paths P/Q sharing ordered start and end; curved oracle produces a nonzero difference, endpoint-only comparator gives zero difference.
4. **SO(2) rotation:** rotate vertices by fixed angles, preserve synthetic T within 1e-12 absolute tolerance.
5. **Genuinely linear null:** k=0 implies T=0 for every path; endpoint-linear comparator exactly explains the specified linear outcome. A method reporting a curvature advantage here fails.
6. **Dose scaling:** a ∈ {0,0.5,1,-1} has exact prescribed signed scaling.

This is a deliberately constructed field with known curvature. Passing tests certifies only arithmetic and test harness behavior, not manifold identification in a real model.

## Future study contract, to freeze before accessing real outcomes

- **Unit and provenance:** independent history seed × held-out probe × model revision, with checkpoint/tokenizer hashes and predeclared layer/activation site. Cluster inference at history-seed level, never token level.
- **Design:** frozen ordered path pairs with common endpoints; independent training paths for any fitted representation; held-out paths, seeds, probes and model revisions for confirmation. Compare both path orders and reversals. Fix basis transformations and apply them consistently to state, transport and decoder.
- **Controls:** dose 0; sign-reversed intervention; path permutations; endpoint-matched linear and equal-parameter nonlinear baselines; norm-matched orthogonal directions; sham intervention; negative-control probes; matched activation magnitude, computation and tuning budgets. Hold trajectory lengths and token positions constant where feasible.
- **Primary estimand:** seed-clustered difference-in-differences of held-out, independently scored persona-related behavioral readout between P and Q, minus the matched endpoint-linear comparator, at the preregistered nonzero dose. Report the sign, units, uncertainty and all null/negative-control outcomes. Behavioral changes alone do not identify a curved latent geometry.
- **Geometric secondary:** estimate path-dependent transport on held-out activation trajectories, with independent train/test separation; quantify sensitivity to coordinate choice, gauge conventions, basis changes and decoder readout. Predefine metric, estimator, dimensionality, regularization and alignment procedure.
- **Statistical plan:** freeze one primary dose, layer, sample size, smallest effect of interest and alpha before data collection; use history-cluster bootstrap CIs and seed-level randomization or sign-flip tests where exchangeability holds. Holm-adjust multiple secondary tests, report raw effects and uncertainty. Decide power using an external pilot, never synthetic oracle effects.
- **Decision rule:** claim a measured path-dependent behavioral effect only if the preregistered primary comparison passes its frozen threshold with null checks acceptable. Claim neural geometric holonomy only with additional evidence of a connection, transport, nontrivial closed-loop behavior, representation/gauge consistency and independent replication. Otherwise classify unknown or unsupported; a null result stays a null result.
- **Data and reproducibility:** commit the frozen protocol/seed manifest before model evaluation; retain raw event/activation provenance and code SHA, blinded scoring instructions, exclusions, attrition, all tested cells and environment hashes. Keep pilot and confirmatory analyses distinct.

## Evidence limits

The oracle assumes a 2D Euclidean geometry and constant curvature, so its positive case is guaranteed by construction. Rotation tests apply to proper orthogonal coordinate changes with the connection transformed consistently. This package cannot establish whether neural persona effects occupy any manifold, whether transport is causal, whether latent state is identifiable, whether any learned connection exists, or whether a model exhibits persona persistence. No model or GPU evaluation was performed.
