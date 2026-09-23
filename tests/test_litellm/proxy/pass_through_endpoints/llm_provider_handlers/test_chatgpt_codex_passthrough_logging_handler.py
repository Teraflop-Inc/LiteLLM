import json
from unittest.mock import MagicMock

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.chatgpt_codex_passthrough_logging_handler import (
    ChatGPTCodexPassthroughLoggingHandler,
    request_messages,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _sse(*events):
    return [f"data: {json.dumps(e)}" for e in events]


def _logging_obj():
    obj = MagicMock()
    obj.litellm_call_id = "call-1"
    obj.model_call_details = {}
    return obj


USAGE = {
    "input_tokens": 100,
    "output_tokens": 7,
    "total_tokens": 107,
    "input_tokens_details": {"cached_tokens": 80},
    "output_tokens_details": {"reasoning_tokens": 3},
}


def test_endpoint_type_is_chatgpt_codex():
    assert (
        HttpPassThroughEndpointHelpers.get_endpoint_type(
            "https://chatgpt.com/backend-api/codex/responses"
        )
        == EndpointType.CHATGPT_CODEX
    )
    assert (
        HttpPassThroughEndpointHelpers.get_endpoint_type("https://chatgpt.com/other")
        == EndpointType.GENERIC
    )


def test_output_rebuilt_from_output_item_done_when_completed_is_empty():
    # The ChatGPT backend sends response.completed with an empty output (store=false).
    chunks = _sse(
        {"type": "response.created", "response": {"model": "gpt-5.5"}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
            },
        },
        {"type": "response.completed", "response": {"model": "gpt-5.5", "output": [], "usage": USAGE}},
    )
    obj = _logging_obj()
    out = ChatGPTCodexPassthroughLoggingHandler.handle_collected_chunks(
        litellm_logging_obj=obj,
        request_body={"model": "gpt-5.5", "input": [{"role": "user", "content": "hi"}]},
        all_chunks=chunks,
    )
    result = out["result"]
    assert result.model == "gpt-5.5"
    call = result.choices[0].message.tool_calls[0]
    assert call.function.name == "exec_command"
    assert result.choices[0].finish_reason == "tool_calls"
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 7
    assert result.usage.prompt_tokens_details.cached_tokens == 80
    assert obj.model_call_details["model"] == "gpt-5.5"
    assert obj.model_call_details["messages"] == [{"role": "user", "content": "hi"}]


def test_text_output_from_completed():
    chunks = _sse(
        {
            "type": "response.completed",
            "response": {
                "model": "gpt-5.5",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "done"}]}
                ],
                "usage": USAGE,
            },
        }
    )
    out = ChatGPTCodexPassthroughLoggingHandler.handle_collected_chunks(
        litellm_logging_obj=_logging_obj(), request_body={}, all_chunks=chunks
    )
    assert out["result"].choices[0].message.content == "done"
    assert out["result"].choices[0].finish_reason == "stop"


def test_no_completed_event_returns_none():
    out = ChatGPTCodexPassthroughLoggingHandler.handle_collected_chunks(
        litellm_logging_obj=_logging_obj(),
        request_body={},
        all_chunks=["data: not json", "data: [DONE]"],
    )
    assert out["result"] is None


def test_request_messages_maps_responses_input():
    msgs = request_messages(
        {
            "instructions": "be brief",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ls"}]},
                {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a\nb"},
            ],
        }
    )
    assert [m["role"] for m in msgs] == ["system", "system", "user", "assistant", "tool"]
    assert msgs[3]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert msgs[4]["content"] == "a\nb"
