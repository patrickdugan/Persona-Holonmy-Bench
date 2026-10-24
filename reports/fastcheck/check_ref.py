"""Which bf16 scorer is closer to float32? Reference: float32 per-completion logprob."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_holonomy.interventions import STYLE_COMPLETIONS, CausalProbe
from persona_holonomy.protocol import ALL_PROBES, build_cases
from persona_holonomy.targets import COPY_FREE_TARGETS, REGISTERS, copyable_target

model_path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.float32, local_files_only=True).to("cuda").eval()
kw = dict(layer="mid", device="cuda", max_total_tokens=2048,
          chat_template_kwargs={"enable_thinking": False})
cases = [c.to_dict() for c in build_cases(
    repetitions=2, seed=9290, variation="seeded_v1", genres=("gothic", "noir"),
    conditions=("ordered", "lexical_match"), phases=("exit",),
    probes=tuple(p for p in ALL_PROBES if p[0] in ("stale_cache", "log_rotation")))]
items = []
for case in cases:
    pid, g = case["probe_id"], case["genre"]
    texts = ([STYLE_COMPLETIONS[pid]["plain"], STYLE_COMPLETIONS[pid]["theatrical"]]
             + [COPY_FREE_TARGETS[pid][r] for r in REGISTERS] + [copyable_target(g, pid)])
    items.append((case["messages"], texts))
runner = CausalProbe(model, tok, **kw)
ref = [[runner.logprob(m, x) for x in texts] for m, texts in items]
model.to(torch.bfloat16)
runner = CausalProbe(model, tok, **kw)
old = [[runner.logprob(m, x) for x in texts] for m, texts in items]
new = [runner.score_continuations(m, texts) for m, texts in items]


def err(est, field):
    diffs = [abs(a[field] - b[field]) for r, e in zip(ref, est) for a, b in zip(r, e)
             if a[field] is not None]
    return max(diffs), sum(diffs) / len(diffs)


def margin_err(est):
    out = []
    for r, e in zip(ref, est):
        for i in range(1, len(r)):
            out.append(abs((e[i][0] - e[0][0]) - (r[i][0] - r[0][0])))
    return max(out), sum(out) / len(out)


for name, est in (("old_bf16", old), ("new_bf16", new)):
    print(name, "mean(max,avg)=%.3f,%.3f" % err(est, 0), "later=%.3f,%.3f" % err(est, 3),
          "margins=%.3f,%.3f" % margin_err(est))
