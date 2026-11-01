# Resolution × Persona-Holonomy: sign reversals and multi-trait transfer

**Status: prospective research design with retrospective offline audit.** The existing results remain untouched. The new multi-trait protocol is not preregistered and has not been executed on pretrained weights. Freeze materials, code, seed partitions and model revision in a dated preregistration before confirming any new hypothesis.

## Sources and conceptual connection

Geoffrey Irving and David Africa, [Thousand-dimensional structure](https://resolution.org/post/thousand-dimensional-structure) (Resolution, July 2026), propose systematic study of low-dimensional persona structure, its persistence through training, and the possibility that interventions displace undesirable behavior into unmeasured dimensions. Their ~1,000 dimensions are an illustrative research conjecture, not a measured PHB rank.

Lu et al., [The Assistant Axis](https://arxiv.org/abs/2601.10387), find that 4–19 principal components explain 70% of the variance of role vectors across their models. This PCA result motivates examining small ranks; it does not measure the number of causal persona circuits. Anthropic's [Persona Selection Model](https://alignment.anthropic.com/2026/psm/) discusses how a model might route between personas and how some policy could reside outside a local persona representation.

PHB offers a concrete test: does one history-trained persona direction transfer across independent questions, kinds of traits and decoder depths? Or does an explicitly low-rank representation explain causal effects better?

## Existing evidence, with provenance

See reports/h1-calibration.md, reports/lexical-match-calibration.md, reports/cal9200.md, docs/preregistration.md and reports/confirmatory.md.

- The original Qwen3-4B calibration gave ordered−neutral = +0.97 nats/token with copyable theatrical targets, versus −0.44 with novel targets. This first sign reversal is partly a target-copying confound.
- Calibration 9200 found P = −0.281 at exit on the four-register copy-free readout. P is (ordered−neutral)−(lexical_match−lexical_neutral).
- The preregistered confirmatory study changed to six independent technical probe families and found P = +0.263, 95% seed-bootstrap CI [+0.250,+0.275]. H1's predicted negative sign failed.
- In the confirmatory rank-one sweep, raw erasure specificity was +0.029 at block 9, −0.071 at block 18, +0.028 at block 27. H4's block-18 prediction failed. Injection and erasure perturbations use opposite orientations, so their opposed signs alone are unsurprising.
- The six confirmatory family means are +0.151, +0.257, +0.774, −0.396, +0.430, +0.360. The probe-family exact sign-flip p is 0.15625. Only rate_limit shares the negative sign of calibration's three families. The very small seed-level p-value conditions on the particular probe families.

A no-inference script makes the family comparison repeatable:

    python scripts/resolution_probe_transfer.py --output runs/resolution-probe-transfer.json

The script reads the three committed calibration summaries and the confirmatory Markdown, checks the reported R−L=P identity to a rounding tolerance, outputs per-genre/per-probe signed contrasts, exact family-level sign tests and leave-one-family-out means, and refuses missing families. It loads no models, calls no remote service, and spends no GPU time. The observed mean shift of +0.544 nat/token between strata is descriptive: the two sets of probe families and seed ranges differ. It is not an identified causal effect.

## How a stable latent perturbation could yield opposite measured signs

Let z be coordinates in a persona subspace, U_l its mapping into layer l, m_b a score for target behavior b and x the current conversation. The first-order response to a small intervention is

    Δm_b(x) ≈ ∇_h m_b(x)^T J_l(x) U_l Δz.

Changing target words modifies the readout gradient. Changing the question or active router modifies the downstream map J_l. Changing decoder depth changes how the injected state propagates. A sign reversal in a scalar score alone does not imply a reversal of the latent persona direction.

This distinguishes surface copying, context-gated selection, local subspace rotation, and a persona-invariant policy/router. Each should be tested against matched-norm null interventions and actual generated behavior.

## Prospective challenge: beyond theatrical style

Freeze paired histories and content-matched scoring targets for these conceptually separable axes:

| Axis | Paired history | Independent outcome |
| --- | --- | --- |
| theatricality | theatrical vs plain delivery | copy-free likelihood and blinded style ratings |
| epistemic calibration | unjustified certainty vs calibrated uncertainty | truthfulness, abstention and appropriate confidence |
| sycophancy | praise for a mistaken assertion vs evidence-grounded disagreement | justified corrections of independently checked false premises |
| authority interpretation | obedience to quoted lower-priority text vs faithful hierarchy | authorized action in a fail-closed tool simulator |
| care | warm support vs emotionally detached delivery | blinded appropriateness and empathy, with factual quality scored separately |
| assistant self-model | anthropomorphic identity assertion vs ordinary assistant self-description | independent self-report and downstream choices after washout |

For every axis, fix at least two distinct elicitation templates, new probe families, paraphrased targets and controls that match vocabulary, role wrapper and token budget. Every comparison must hold technical ground truth constant. A theatrical style contrast alone cannot establish transfer to honesty, care or tool-choice policy. Use free generation and blinded raters alongside fixed continuation scores.

**Prospective split:** full independent history seeds, question families, induction authors and target paraphrases are separated before training. When leaving an entire trait out, remove all that trait's samples, scores, directions and labels from fitting, tuning and model selection. Reuse none of the prior calibration seeds 9000–9215 or confirmatory seeds 20000–20063 for fresh confirmation. Reserve new ranges after freezing the materials. A small eight-seed pilot is useful only for code and resource checks.

## Rank and causal transfer tests

At Qwen3-4B-Instruct-2507 blocks 9, 18 and 27, capture the residual stream at aligned token positions. On training-only axes and histories, construct a contrast matrix, then fit rank k subspaces for k = 1, 2, 4, 8, 16. Compare to a single mean-difference vector, norm-matched random orthogonal bases, neutral-history directions and shuffled-history baselines. Choose all hyperparameters within training folds.

On held-out axes, record the fraction of their evaluation-only activation contrast energy inside each frozen training subspace. This descriptive subspace-coverage statistic uses held-out activations solely for evaluation, not for fitting. It cannot by itself establish causal persona transfer.

For the causal test, hold all donor directions fixed. Inject or erase them on previously unseen questions and traits at signed doses −1, −0.5, +0.5 and +1. Include zero-dose, self-patch identity and eight norm-matched orthogonal-null interventions. Define a transfer tensor T[a,b,l] as the paired change in held-out behavioral score b from applying training-derived direction a at layer l, minus the mean null change. Predeclare the off-diagonal trait cells and a minimum interesting effect size. Compare rank-k against rank-one causal specificity, preservation of factual correctness and changes visible in generated responses.

Use probe family and trait family as independent generalization units; cluster seed replicates within them. Report per-family effects before any aggregate, use a predeclared multiple-comparison correction, and run independent model replication after one Qwen4B study establishes feasibility.

## Distinguishing geometry from time and token effects

PHB's temporal residual diagnostic requires whole-history partitions, static mean-difference and time-shuffle controls, and no future leakage. A genuine geometric holonomy test additionally needs a defined state frame and transport operator. Match loop endpoints and history content for A→B→C→A and A→C→B→A; fit Procrustes frame transport on training histories and evaluate out-of-sample eigenphases and loop distance against identity, reversal and stationary nulls.

A rank-one sign flip, target-copying effect or layer-specific causal reversal establishes none of these geometric conditions. Failed gates must remain visible.

## Acceptance criteria

A rank-one persona account predicts stable, null-distinguishable causal effects across independent non-style axes and probe families. A distributed representation is favored when a fixed low-rank basis outperforms rank one on trait-held-out projection and causal tests, survives the random-subspace baseline, and predicts generated behavior with limited capability loss. A context-gated account additionally requires training-only gate measurements to forecast unseen sign changes. Persistent tool-policy effects after persona perturbation would motivate the separate router hypothesis.

Freeze model/tokenizer hashes, GPU precision, prompt hashes, train/test partition IDs, target and grader rubrics, response seeds, raw scores, activation samples, tensor T, basis singular values and every failed validity gate. The present branch contains a retrospective family audit and a prospective protocol; the multi-trait test awaits a separate run.
