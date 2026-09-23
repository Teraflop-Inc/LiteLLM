"""
Logging for Codex signed in with ChatGPT, through the `/teraflop-codex` pass-through (ENG2-402).

Codex streams the Responses API from `chatgpt.com/backend-api/codex/responses`. The generic
pass-through logs such a call as model "unknown" with zero tokens and no content, because it
cannot read the stream. This reads the one event that carries everything, the final
`response.completed`: model, output items and usage (input, cached, output, reasoning). It turns
that into a ModelResponse so the normal callbacks (arize_phoenix, otel) log it like any other
LLM call, and turns the request's `input` items into chat messages so the prompt is logged too.

Never logs the bearer: only the request body and the streamed response are read.
"""

import json
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)

COMPLETED_EVENTS = ("response.completed", "response.done")


def _events(all_chunks: List[str]) -> List[dict]:
    """The JSON payloads of the `data:` lines, in order. Non-JSON lines are skipped."""
    out: List[dict] = []
    for line in all_chunks:
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = [
        p.get("text") or ""
        for p in content or []
        if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")
    ]
    return "\n".join(x for x in parts if x)


def request_messages(request_body: dict) -> List[dict]:
    """The Responses API request as chat messages, so the prompt lands as input.value."""
    messages: List[dict] = []
    if request_body.get("instructions"):
        messages.append({"role": "system", "content": str(request_body["instructions"])})
    for item in request_body.get("input") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role") or "user"
            messages.append(
                {"role": "system" if role == "developer" else role, "content": _text(item.get("content"))}
            )
        elif kind in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": output if isinstance(output, str) else json.dumps(output),
                }
            )
        elif kind in ("function_call", "custom_tool_call"):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments", item.get("input", "")),
                            },
                        }
                    ],
                }
            )
    return messages


def completed_to_model_response(response: dict) -> ModelResponse:
    """A `response.completed` payload's `response` object as a chat ModelResponse."""
    text_parts: List[str] = []
    tool_calls: List[ChatCompletionMessageToolCall] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            text_parts.append(_text(item.get("content")))
        elif kind in ("function_call", "custom_tool_call"):
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=item.get("call_id") or item.get("id"),
                    type="function",
                    function=Function(
                        name=item.get("name") or kind,
                        arguments=item.get("arguments", item.get("input", "")) or "",
                    ),
                )
            )
        elif kind == "local_shell_call":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=item.get("call_id") or item.get("id"),
                    type="function",
                    function=Function(name="local_shell", arguments=json.dumps(item.get("action") or {})),
                )
            )
        elif kind == "web_search_call":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=item.get("id"),
                    type="function",
                    function=Function(name="web_search", arguments=json.dumps(item.get("action") or {})),
                )
            )

    usage = response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)

    message = Message(
        content="\n".join(x for x in text_parts if x) or None,
        role="assistant",
        tool_calls=tool_calls or None,
    )
    return ModelResponse(
        model=response.get("model"),
        choices=[
            Choices(
                index=0,
                message=message,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=int(usage.get("total_tokens") or input_tokens + output_tokens),
            prompt_tokens_details={"cached_tokens": cached},
            completion_tokens_details={"reasoning_tokens": reasoning},
        ),
    )


class ChatGPTCodexPassthroughLoggingHandler:
    @staticmethod
    def handle_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
        all_chunks: List[str],
    ) -> Dict[str, Any]:
        """Returns {"result": ModelResponse | None, "kwargs": dict} like the other handlers."""
        kwargs: Dict[str, Any] = {}
        events = _events(all_chunks)
        completed: Optional[dict] = next(
            (e.get("response") for e in reversed(events) if e.get("type") in COMPLETED_EVENTS),
            None,
        )
        if not isinstance(completed, dict):
            verbose_proxy_logger.warning(
                "[codex-passthrough] no response.completed in %d events; logging raw", len(events)
            )
            return {"result": None, "kwargs": kwargs}

        if not completed.get("output"):
            # The ChatGPT backend (store=false) sends `response.completed` with an empty
            # `output`; the items only arrive as `response.output_item.done` events.
            completed = {**completed, "output": [
                e["item"] for e in events
                if e.get("type") == "response.output_item.done" and isinstance(e.get("item"), dict)
            ]}
        model_response = completed_to_model_response(completed)
        model = completed.get("model") or request_body.get("model") or "unknown"
        model_response.id = litellm_logging_obj.litellm_call_id
        model_response.model = model

        details = litellm_logging_obj.model_call_details
        details["model"] = model
        details["custom_llm_provider"] = "openai"
        messages = request_messages(request_body)
        if messages:
            details["messages"] = messages
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = "openai"

        usage = model_response.usage  # type: ignore[attr-defined]
        verbose_proxy_logger.debug(
            "[codex-passthrough] model=%s input=%s output=%s messages=%d tool_calls=%d",
            model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            len(messages),
            len(model_response.choices[0].message.tool_calls or []),  # type: ignore[union-attr]
        )
        return {"result": model_response, "kwargs": kwargs}
