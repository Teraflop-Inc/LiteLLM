"""ENG2-1561: the arize_phoenix logger must write the response-side detail the
OpenInference semconv already has slots for: reasoning tokens, prompt cache read and
write, and the finish reason. Before this, only the three token totals were set and
everything else lived (or not) in metadata.usage_object. No network: a fake span
records every attribute `set_attributes` writes."""
from litellm.integrations._types.open_inference import SpanAttributes
from litellm.integrations.arize._utils import set_attributes


class _Span:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v

    def record_exception(self, e):  # pragma: no cover - only on failure
        self.attrs["__exception__"] = repr(e)


def _kwargs():
    return {
        "standard_logging_object": {
            "call_type": "acompletion",
            "model": "claude-opus-5",
            "metadata": {},
            "messages": [{"role": "user", "content": "hi"}],
        },
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {},
        "litellm_params": {"metadata": {}},
    }


def test_logger_writes_reasoning_cache_and_finish_reason():
    span = _Span()
    response_obj = {
        "id": "msg_1",
        "model": "claude-opus-5",
        "choices": [{"finish_reason": "end_turn", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "total_tokens": 160,
            "completion_tokens_details": {"reasoning_tokens": 17},
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 20,
        },
    }
    set_attributes(span, _kwargs(), response_obj)
    assert "__exception__" not in span.attrs, span.attrs.get("__exception__")
    assert span.attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 17
    assert span.attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 100
    assert span.attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] == 20
    assert span.attrs["llm.response.finish_reason"] == "end_turn"


def test_logger_is_quiet_when_the_detail_is_absent():
    span = _Span()
    response_obj = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    set_attributes(span, _kwargs(), response_obj)
    assert "__exception__" not in span.attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING not in span.attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in span.attrs
    assert "llm.response.finish_reason" not in span.attrs
