"""/v1/messages must forward the request fields Claude Code sends.

ENG2-1773: the fork's allowlist silently dropped `output_config` (the effort setting) and
`context_management` from every proxied Claude Code request. The body below is the field
set Claude Code 2.1.321 sent, captured against a fake backend on 2026-11-09.
"""
import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.utils import (
    AnthropicMessagesRequestUtils,
)

CLAUDE_CODE_FIELDS = {
    "max_tokens": 32000,
    "metadata": {"user_id": "u"},
    "stream": True,
    "system": [{"type": "text", "text": "s"}],
    "thinking": {"type": "adaptive", "display": "omitted"},
    "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    "output_config": {"effort": "medium"},
}

NEWER_MESSAGES_FIELDS = {
    "container": "container_abc",
    "service_tier": "auto",
    "thread": {"id": "t1"},
}


@pytest.mark.parametrize("field", sorted({**CLAUDE_CODE_FIELDS, **NEWER_MESSAGES_FIELDS}))
def test_messages_field_is_forwarded(field):
    params = {**CLAUDE_CODE_FIELDS, **NEWER_MESSAGES_FIELDS}
    kept = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(params)
    assert kept.get(field) == params[field], f"/v1/messages dropped `{field}`"
