"""Shared chat-template rendering with optional model-specific arguments."""
from __future__ import annotations


def render_chat(tokenizer, messages, *, chat_template_kwargs: dict | None = None,
                **kwargs):
    """Preserve tokenizer defaults unless template arguments are supplied."""
    return tokenizer.apply_chat_template(
        messages, **kwargs, **(chat_template_kwargs or {}))
