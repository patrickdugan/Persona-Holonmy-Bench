"""Real-model check: all_margins (prefix-cached, batched) vs per-completion logprob."""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_holonomy.interventions import STYLE_COMPLETIONS, CausalProbe
from persona_holonomy.protocol import ALL_PROBES, build_cases
from persona_holonomy.targets import COPY_FREE_TARGETS, REGISTERS, copyable_target

model_path, dtype = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=getattr(torch, dtype), local_files_only=True).to("cuda").eval()
runner = CausalProbe(model, tok, layer="mid", device="cuda", max_total_tokens=2048,
                     chat_template_kwargs={"enable_thinking": False})
cases = [c.to_dict() for c in build_cases(
    repetitions=2, seed=9290, variation="seeded_v1", genres=("gothic", "noir"),
    conditions=("ordered", "lexical_match"), phases=("exit",),
    probes=tuple(p for p in ALL_PROBES if p[0] in ("stale_cache", "log_rotation")))]
worst, t_old, t_new = 0.0, 0.0, 0.0
d_mean = d_first = d_tail = d_margin = 0.0
for case in cases:
    m, pid, g = case["messages"], case["probe_id"], case["genre"]
    texts = ([STYLE_COMPLETIONS[pid]["plain"], STYLE_COMPLETIONS[pid]["theatrical"]]
             + [COPY_FREE_TARGETS[pid][r] for r in REGISTERS] + [copyable_target(g, pid)])
    torch.cuda.synchronize(); t = time.time()
    old = [runner.logprob(m, x) for x in texts]
    torch.cuda.synchronize(); t_old += time.time() - t; t = time.time()
    new = runner.score_continuations(m, texts)
    torch.cuda.synchronize(); t_new += time.time() - t
    for a, b in zip(old, new):
        d_mean = max(d_mean, abs(a[0] - b[0])); d_first = max(d_first, abs(a[2] - b[2]))
        if a[3] is not None:
            d_tail = max(d_tail, abs(a[3] - b[3]))
    am = runner.all_margins(m, probe_id=pid, genre=g)
    old_cf = sum(old[i][0] for i in range(2, 6)) / 4 - old[0][0]
    d_margin = max(d_margin, abs(am["margin"] - (old[1][0] - old[0][0])),
                   abs(am["copy_free_margin"] - old_cf))
print(f"{dtype} cases={len(cases)} mean={d_mean:.1e} first={d_first:.1e} later={d_tail:.1e} margins={d_margin:.1e} old={t_old:.1f}s new={t_new:.1f}s "
      f"speedup={t_old / t_new:.1f}x")
