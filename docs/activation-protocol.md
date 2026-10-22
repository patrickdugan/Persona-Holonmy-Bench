# Open-weight activation protocol (optional)

`phb capture` is a *forward-only* hidden-state measurement; it never generates an answer, steers activations, or trains a probe. It measures the final input position just before assistant generation after replaying **the complete scripted transcript** in one forward pass per case.

## Model selection and safety

Run only on a locally installed or previously cached model by default. Both downloading model weights (`--allow-download`) and loading custom model-supplied Python (`--trust-remote-code`) require explicit choices. Pin the exact revision of the model and tokenizer, preferably using a local directory, and record software/runtime versions externally.

The `dtype` option uses recent Transformers `from_pretrained(dtype=...)` interfaces. Verify actual loaded parameter dtypes, especially for architectures with composite model configs; requesting fp32 can silently yield a different precision on some Transformers releases. Models with multimodal or custom chat templates may need specialized adapters.

## Example

```bash
pip install -e '.[mechanistic]'
phb generate --repetitions 5 --output runs/cases.jsonl
phb capture --model /path/to/local/HF/chat-model --input runs/cases.jsonl \
  --layers 0,8,last --output runs/activations.jsonl
phb activation-report --input runs/activations.jsonl \
  --output runs/activation-summary.json
```

The default `--max-input-tokens=4096` **fails closed** when a case exceeds the limit. It never truncates the context silently. `--max-cases` is an optional budget cap. For smaller systems, use only `--layers last` and `--max-cases 16` in early tests.

The output includes input-token hashes, token counts, selected full hidden-state vectors, actual first-model-parameter dtype, tokenizer name and case identity. Recording vectors may generate large output files; do not commit data derived from sensitive prompts. Input messages remain in your generated cases file.

## Interpretation

Hidden state 0 is the embedding output in common Hugging Face causal models, while the remaining entries represent model-layer outputs. The selected last input position can be the assistant role marker, based on the tokenizer's template. Because full histories have different lengths, this is **not a controlled identical-token-position intervention**.

`activation-report` computes cosine distances to matched zero-history baselines and to a `tail_matched` trajectory with the same final induction turn and its absolute position. Such differences are associations with text context, and do not identify the causal sparse features, a VPD decomposition, a geometric connection, or a holonomy operator. Establish a transport law and interventions before making holonomy claims.
