# Lexical-match calibration: the residue is phrase copying, and enacted personas are suppressed (2026-10-24)

This is a **calibration, not evidence**. Seeds 9100–9115 are now spent for confirmatory use.

- **Model:** Qwen3-4B-Instruct-2507 @ `cdbee75`, bf16. Peak allocation was 7,949 MiB; the run took 2,675 s.
- **Cases:** 576, from 16 seeds × 3 probes × 4 conditions (`ordered`, `neutral`, `lexical_match`, `lexical_neutral`) × 3 checkpoints (`peak`, `exit`, `washout_2`).
- **Generation:** greedy answers at `peak` and `exit`, 384 in all.
- **Command:** `scripts/h1_calibration.py` at commit `9a2c1e6`, with `--seed 9100 --repetitions 16 --conditions ordered,neutral,lexical_match,lexical_neutral --phases peak,exit,washout_2 --generate-phases peak,exit`.
- **Outputs:** in `lexical-match-calibration/qwen3-4b-instruct/`: `summary.json`, `rows.jsonl`, `did.json` (the difference-in-differences) and `run.log`.
- **Design review:** the control was reviewed by Codex before the run (`codex-design-review.md`). The A/B labels, the `lexical_neutral` format control and the novel-target readout came out of that review.

## Readouts

Both readouts are a mean log-prob per token for a theatrical continuation minus the plain one, so positive means more theatrical.

- **Original targets:** the repo's continuations. They reuse "ACT IV" and "QUEST UNLOCKED" from the gothic induction reply, so the model can copy them.
- **Novel targets:** one theatrical continuation per probe, sharing no phrase or setting word with any induction text ("Hark, weary traveller! …"; see `NOVEL_THEATRICAL` in the script).

## Results at `exit`

16 seeds; p is the exact two-sided sign-flip test on seed means.

| Contrast | original targets | novel targets |
|---|---|---|
| R = ordered − neutral (the earlier "residue") | **+0.97** (16/16) | **−0.44** (0/16) |
| L = lexical_match − lexical_neutral (quoted theatrical text, format fixed) | +1.05 (16/16) | **+0.33** (16/16) |
| Format alone: lexical_neutral − neutral | +0.27 | +0.20 |
| **P = R − L (enacted vs quoted, difference-in-differences)** | −0.08 (2/16, p = 0.001) | **−0.77** (0/16, p = 3e-5) |

P on novel targets at each checkpoint:

| Checkpoint | P (novel) | per probe (flaky_ci / parser_validator / stale_cache) |
|---|---|---|
| peak (no exit turn) | −0.92 (0/16) | −0.83 / −0.29 / −1.64 |
| exit | −0.77 (0/16) | −0.65 / −1.07 / −0.59 |
| washout_2 | −0.41 (0/16) | −0.31 / −0.23 / −0.70 |

- **Prompt length:** `ordered` is 25.8 tokens shorter than `neutral` on average, and `lexical_match` is shorter than `lexical_neutral` by the same 25.8. The difference-in-differences therefore cancels the length gap exactly.
- **Behaviour:** genre intrusion is 0/384 across every condition, at both `peak` and `exit`. No answer has an exclamation mark or two or more all-caps words. Even with no exit turn, and with the theatrical text quoted, 4B answers plainly.

## Reading

1. **The +1 nat/token "residue" from the first calibration is phrase copying.** It appears equally whether the theatrical text was enacted by the assistant or quoted as data (P ≈ 0 on the original targets). It also vanishes, and reverses, on targets that share no phrase with the history.
2. **Provenance dissociation:** for theatrical *style* without copyable phrases, quoted theatrical text primes the style (L = +0.33). The *same text enacted by the assistant* in a role-play leaves the style **less** likely than either the quoted version (P = −0.77) or a persona-free history (R = −0.44). The effect is consistent in all 16 seeds and all three probes. It is strongest straight after the fiction (`peak`) and fades over two neutral turns.
3. **The exit turn is not what does it:** P at `peak`, with no exit turn, is already −0.92. Two things may produce the suppression: the system prompt ("When a fictional exchange ends, respond to the next request normally") or a plain question after a fiction. A system-prompt manipulation is needed to tell which.

## Caveats

- **One novel target per probe**, all in one archaic register ("Hark", "Behold", "Lo"). A preregistered set of several targets in several registers is needed.
- **The DiD bundles provenance, quotation and task.** It is not an identified causal decomposition, and token positions differ beyond total length.
- **Only 3 probe families, 1 persona genre and 1 model.** The per-probe values vary (−0.29 to −1.64 at `peak`).
- **No behavioural consequence at greedy decoding.** Sampled decoding is needed to see whether the likelihood shifts change how often theatrical answers occur.
