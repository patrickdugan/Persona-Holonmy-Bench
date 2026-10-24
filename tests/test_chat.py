"""Template options stay opt-in and reach the tokenizer unchanged."""
import pytest

from persona_holonomy.chat import render_chat
from persona_holonomy.interventions import CausalProbe


@pytest.mark.parametrize("options", [None, {}, {"enable_thinking": False}])
def test_render_chat_forwards_only_requested_kwargs(options):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.kwargs = messages, kwargs
            return result

    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "Hello"}]
    result = object()
    original_kwargs = dict(tokenize=True, add_generation_prompt=True,
                           return_tensors="pt", return_dict=True)
    assert render_chat(tokenizer, messages, chat_template_kwargs=options,
                       **original_kwargs) is result
    assert tokenizer.messages is messages
    assert tokenizer.kwargs == {**original_kwargs, **(options or {})}


def test_render_chat_omitted_options_add_no_kwargs():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == []
            assert kwargs == {}
            return "rendered"

    assert render_chat(Tokenizer(), []) == "rendered"


@pytest.mark.parametrize("options", [None, {"enable_thinking": False}])
def test_causal_probe_forwards_template_kwargs(options):
    torch = pytest.importorskip("torch")
    from test_interventions import toy_model

    model, tokenizer = toy_model()
    original = tokenizer.apply_chat_template
    calls = []

    def record(messages, **kwargs):
        calls.append(kwargs)
        return original(messages, **kwargs)

    tokenizer.apply_chat_template = record
    messages = [{"role": "user", "content": "Hello"}]
    probe = CausalProbe(model, tokenizer, chat_template_kwargs=options)
    prefix = probe._prefix(messages)
    assert calls == [{
        "tokenize": True, "add_generation_prompt": True,
        "return_tensors": "pt", "return_dict": True, **(options or {}),
    }]
    assert torch.equal(prefix["input_ids"], original(messages)["input_ids"])
