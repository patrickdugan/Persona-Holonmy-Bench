import pytest
from persona_holonomy.activations import parse_layers, cosine_distance, activation_report


def test_parse_layers_and_bounds():
    assert parse_layers("0,2,last,2", 5) == (0, 2, 4)
    with pytest.raises(ValueError, match="outside"):
        parse_layers("5", 5)
    with pytest.raises(ValueError):
        parse_layers("", 5)


def test_cosine_distance():
    assert cosine_distance([1, 0], [1, 0]) == 0
    assert cosine_distance([1, 0], [0, 1]) == 1
    with pytest.raises(ValueError):
        cosine_distance([0, 0], [1, 0])
    with pytest.raises(ValueError):
        cosine_distance([1, 2], [1])


def _record(cond, phase, vector):
    return {"schema_version": 1, "case_id": f"{cond}.{phase}.x.s1",
            "condition": cond, "phase": phase, "seed": 1, "probe_id": "x",
            "model": "local-test", "tokenizer": "tok", "vectors": {"hs_2": vector}}


def test_activation_report_matched_pairs():
    rows = [_record("ordered", "baseline", [1., 0.]),
            _record("tail_matched", "baseline", [1., 0.]),
            _record("ordered", "exit", [0., 1.]),
            _record("tail_matched", "exit", [1., 0.])]
    report = activation_report(rows)
    assert report["n_records"] == 4
    comparison = [c for c in report["comparisons"] if c["metric"] == "tail_matched_distance"
                  and c["phase"] == "exit"]
    assert comparison[0]["mean_cosine_distance"] == 1
    with pytest.raises(ValueError, match="different models"):
        activation_report(rows + [{**rows[0], "model": "other", "case_id": "other"}])
    with pytest.raises(ValueError, match="Duplicate"):
        activation_report(rows + [rows[0]])


def test_capture_with_fake_hf_model_is_offline_and_records_last_token(tmp_path, monkeypatch):
    """Exercises the optional adapter plumbing without large model downloads."""
    import json
    import sys
    from types import SimpleNamespace
    from persona_holonomy.activations import capture

    class Tensor:
        def __init__(self, value):
            self.value = value
            self.shape = (len(value), len(value[0])) if isinstance(value[0], list) else (len(value),)
        def to(self, device): return self
        def __getitem__(self, indices):
            if indices == (0, -1):
                return Tensor(self.value[0][-1])
            raise AssertionError(f"unexpected index {indices}")
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def tolist(self): return self.value

    class FakeTokenizer:
        name_or_path = "fake-chat-tokenizer"
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return FakeTokenizer()
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] and kwargs["add_generation_prompt"]
            return {"input_ids": Tensor([[12, 13, 14]])}

    class FakeModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert kwargs["dtype"] == "auto"
            assert kwargs["local_files_only"]
            return FakeModel()
        def to(self, device): return self
        def eval(self): return self
        def parameters(self): yield SimpleNamespace(dtype="float32")
        def __call__(self, **kwargs):
            assert kwargs["output_hidden_states"] and kwargs["use_cache"] is False
            return SimpleNamespace(hidden_states=(
                Tensor([[[1., 0.], [1., 0.], [1., 0.]]]),
                Tensor([[[0., 1.], [0., 1.], [0., 1.]]]),
            ))

    class InferenceMode:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    fake_torch = SimpleNamespace(
        Tensor=Tensor, cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        inference_mode=lambda: InferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeModel
    ))
    src = tmp_path / "cases.jsonl"
    dst = tmp_path / "acts.jsonl"
    src.write_text(json.dumps({
        "case_id": "ordered.exit.x.s1", "condition": "ordered",
        "phase": "exit", "probe_id": "x", "seed": 1,
        "messages": [{"role": "user", "content": "hello"}]
    }) + "\n")
    count = capture(input_path=str(src), output_path=str(dst),
                    model_name="fake-local-model", layers="0,last")
    assert count == 1
    record = json.loads(dst.read_text())
    assert record["input_ids_sha256"]
    assert record["vectors"] == {"hs_0": [1., 0.], "hs_1": [0., 1.]}
    assert record["token_count"] == 3
    with pytest.raises(ValueError, match="limit is 2"):
        capture(input_path=str(src), output_path=str(dst), model_name="fake-local-model",
                max_input_tokens=2)
