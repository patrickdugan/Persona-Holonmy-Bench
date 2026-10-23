# H1 calibration: is there persona carryover to explain? (2026-10-23)

This is a **calibration, not evidence**. It was run before any preregistration, on seeds 9000–9015, which no pilot or confirmatory plan uses. It tests the premise of the mechanism study. After a persona history and an exit instruction, does the model treat an unrelated technical question more theatrically, and does the `ordered` history do so more than its controls?

**Short answer: no behavioural carryover in either model, and the study's main contrast (`ordered` − `tail_matched`) is small and runs the wrong way.** Only the contrast against a persona-free `neutral` history is large. That difference is a likelihood shift with no visible change in the answers.

## Setup

| | Qwen3-4B-Instruct-2507 | Qwen3-0.6B |
|---|---|---|
| Revision | `cdbee75f17c01a7cc42f958dc650907174af0554` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Verification | every file matches the upstream LFS SHA-256 or git blob at that revision | same (see `qwen3-0.6b-pilot.md`) |
| Chat mode | non-thinking only; the prompt ends at `<|im_start|>assistant\n` | hybrid thinking; greedy answers open with `<think>` |
| dtype | bfloat16 (the native dtype) | bfloat16 |
| Runtime / peak allocated | 2,557 s / 7,900 MiB (11 GB vramorama lease) | 938 s / 1,310 MiB |

Shared settings:
- **Cases:** 16 seeds (9000–9015) × 4 conditions (`ordered`, `tail_matched`, `neutral`, `permuted`) × 3 checkpoints (`baseline`, `exit`, `washout_2`) × 3 probes (`flaky_ci`, `parser_validator`, `stale_cache`) = 576 cases per model. The variation is `seeded_v1`.
- **Readouts:**
  - The unpatched forced-continuation style margin: the theatrical continuation's mean log-prob per token minus the plain one's, split into first-token and later-token parts.
  - At `exit`, a greedy 96-token answer scored with `score_response`, the repo's lexical proxy.
- **Statistics:** each contrast is a paired difference averaged over probes within each seed. p-values come from the exact two-sided sign-flip test, over 2^16 sign patterns.
- **Command:** `scripts/h1_calibration.py` (in this commit). Outputs are in `h1-calibration/<model>/` (`summary.json`, `rows.jsonl` including every generated answer, `run.log`).

## Results

At `baseline`, every contrast is exactly 0. The prompts are identical before any history, so this placebo check passes.

**Qwen3-4B-Instruct.** Mean margin difference in log-prob per token, positive = more theatrical, with the number of seeds out of 16 where it was positive:

| Contrast | exit margin | first token | later tokens | washout_2 margin |
|---|---|---|---|---|
| ordered − tail_matched | **−0.063** (2/16, p = 1e-4) | −0.088 | −0.061 | −0.110 (0/16) |
| ordered − permuted | +0.068 (12/16, p = 0.16) | −0.354 | +0.084 | −0.150 (1/16) |
| ordered − neutral | **+0.967** (16/16, p = 3e-5) | −1.142 | +1.061 | +1.116 (16/16) |

**Qwen3-0.6B:**

| Contrast | exit margin | washout_2 margin |
|---|---|---|
| ordered − tail_matched | −0.041 (0/16) | +0.001 (7/16, p = 0.81) |
| ordered − permuted | −0.046 (0/16) | −0.071 (1/16) |
| ordered − neutral | +0.908 (16/16) | +0.921 (16/16) |

**Free generation (exit, 192 greedy answers per model):**
- `genre_intrusion_rate` is **0** in every condition for both models.
- For 4B, no answer contains an exclamation mark or two or more all-caps words. Every answer is a plain technical answer: compare the four conditions for seed 9000 / `flaky_ci` in `rows.jsonl`.
- For 0.6B the 96 tokens are mostly `<think>` reasoning, so its lexical score is uninformative.

**First-token scale (4B):** the theatrical first token (`ACT`) sits about 24 nats below the plain one in every condition. The model essentially never starts an answer in persona after the exit instruction.

## Reading

1. **Behaviour:** after the exit turn, both models answer plainly whatever came before. With this protocol there is no behavioural persona carryover for the mechanism study to explain.
2. **Main contrast (`ordered` vs `tail_matched`, the source and reference used by the sweep and the pilot):** at `exit`, `ordered` is *less* theatrical than `tail_matched` in both models. The difference is small: −0.04 to −0.11 log-prob per token, and about 0 at `washout_2` for 0.6B. The hypothesis that how a persona builds up matters beyond the last turns fails at calibration.
3. **Against `neutral`:** the large, perfectly consistent difference (about +1 log-prob per token, 16/16 seeds, both models, both checkpoints) is a **likelihood residue**. Theatrical continuations become more probable once theatrical text is in the context, but greedy behaviour doesn't change. `tail_matched` and `permuted` contain the same turns and show the same residue, so it most likely reflects content priming (copyable tokens in context), not a persona state.
4. **0.6B pilot confound:** the first-token margins in `qwen3-0.6b-pilot.md` were taken where the model's natural next token is `<think>`. They should not be compared with the 4B numbers or treated as an answer-style readout.

## Implications for the paper spec

As specified, H1 (`ordered` > `tail_matched`, plus behavioural carryover) would fail on both models, which makes H2 and H3 moot. Options, cheapest first:

- **A. Reframe around the residue.** Study why theatrical continuations stay about 1 log-prob per token more likely after a persona history, even though behaviour returns to plain. Ask whether SAE/VPD features carry it, and whether percolation selection finds them. The control must separate content priming from persona state. Use a `neutral` history padded with the same theatrical tokens but no persona framing (the "lexical-match" condition, which needs a new protocol condition).
- **B. Make carryover happen first.** A weaker or missing exit instruction, sampled rather than greedy decoding, or style-inviting probes. Then re-run this calibration on fresh seeds before any preregistration.
- **C. Publish the negative.** A short note: transcript-order persona carryover is absent in Qwen3 0.6B and 4B under explicit exit instructions, with a likelihood residue but no behavioural effect.

Whichever option is chosen, preregister it with fresh seeds. These calibration seeds are now spent for confirmatory use.
