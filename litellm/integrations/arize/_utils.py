import ast
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.integrations._types.open_inference import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.utils import StandardLoggingPayload

from opentelemetry import trace

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span, Tracer as _Tracer

    Span = Union[_Span, Any]
    Tracer = Union[_Tracer, Any]
else:
    Span = Any
    Tracer = Any

LITELLM_TRACER_NAME = os.getenv("OTEL_TRACER_NAME", "litellm")


def cast_as_primitive_value_type(value) -> Union[str, bool, int, float]:
    """
    Converts a value to an OTEL-supported primitive for Arize/Phoenix observability.
    """
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def _usage_field(obj: Any, key: str) -> Any:
    """Read `key` off a usage-shaped value whether it is a dict or a pydantic object.
    The response object reaches this logger as a dict, but its nested usage details
    may still be wrappers depending on the provider path. None in, None out."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def safe_set_attribute(span: Span, key: str, value: Any):
    """
    Sets a span attribute safely with OTEL-compliant primitive typing for Arize/Phoenix.
    """
    primitive_value = cast_as_primitive_value_type(value)
    span.set_attribute(key, primitive_value)


def try_parse_json(value: str) -> Any:
    """
    Tries parsing a string as a JSON object.
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def try_parse_literal(value: str) -> Any:
    """
    Tries parsing a string as a Python literal.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def to_json_string(obj: Any) -> str:
    """
    Safely converts a Python object to JSON string.
    """
    try:
        return safe_dumps(obj)
    except (TypeError, ValueError):
        return str(obj)


def detect_mimetype_and_serialize(value: Any) -> Tuple[str, str]:
    """
    Detects MIME type for a given value & returns stringified value with MIME type for Arize/Phoenix.
    """
    if isinstance(value, (dict, list)):
        return to_json_string(value), OpenInferenceMimeTypeValues.JSON.value

    if isinstance(value, str):
        json_like_chars = ('{', '}', '[', ']', ':')
        if any(c in value for c in json_like_chars):
            parsed = try_parse_json(value)
            if isinstance(parsed, (dict, list)):
                return to_json_string(parsed), OpenInferenceMimeTypeValues.JSON.value

        parsed = try_parse_literal(value)
        if isinstance(parsed, (dict, list)):
            return to_json_string(parsed), OpenInferenceMimeTypeValues.JSON.value

    return str(value), OpenInferenceMimeTypeValues.TEXT.value


def create_anthropic_llm_child_span(
    tracer: Tracer,
    parent_span: Span,
    span_name: str,
    attributes: Dict[str, Any],
    span_kind: str,
    model_name: Optional[str] = None,
) -> None:
    """
    Creates a child span with OpenInference-compliant LLM attributes for Anthropic LLM.
    """
    with tracer.start_as_current_span(
        span_name, context=trace.set_span_in_context(parent_span)
    ) as child_span:
        for key, value in attributes.items():
            safe_set_attribute(child_span, key, value)
        safe_set_attribute(
            child_span,
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            span_kind,
        )
        if span_kind == OpenInferenceSpanKindValues.LLM.value:
            safe_set_attribute(
                child_span,
                SpanAttributes.LLM_PROVIDER,
                OpenInferenceLLMProviderValues.ANTHROPIC.value,
            )
            safe_set_attribute(
                child_span,
                SpanAttributes.LLM_SYSTEM,
                OpenInferenceLLMSystemValues.ANTHROPIC.value,
            )
            if model_name:
                safe_set_attribute(child_span, SpanAttributes.LLM_MODEL_NAME, model_name)


def build_input_output_attributes(
    input_value: str,
    input_mime: str,
    output_value: str,
    output_mime: str,
    input_index: int,
    output_index: int,
) -> Dict[str, Any]:
    """
    Builds OpenInference-compliant LLM attributes dictionary for paired input/output messages.
    """
    return {
        SpanAttributes.INPUT_VALUE: input_value,
        SpanAttributes.INPUT_MIME_TYPE: input_mime,
        SpanAttributes.OUTPUT_VALUE: output_value,
        SpanAttributes.OUTPUT_MIME_TYPE: output_mime,
        "llm.message.input_index": input_index,
        "llm.message.output_index": output_index,
    }


def build_single_message_attributes(
    role: str, index: int, value: str, mime_type: str
) -> Dict[str, Any]:
    """
    Builds OpenInference-compliant LLM attributes for a single user or assistant message.
    """
    return {
        "llm.message.role": role,
        "llm.message.index": index,
        (
            SpanAttributes.INPUT_VALUE
            if role == "user"
            else SpanAttributes.OUTPUT_VALUE
        ): value,
        (
            SpanAttributes.INPUT_MIME_TYPE
            if role == "user"
            else SpanAttributes.OUTPUT_MIME_TYPE
        ): mime_type,
    }


def handle_anthropic_claude_code_tracing(
    tracer: Tracer,
    parent_span: Span,
    kwargs: Dict[str, Any],
    response_obj: Dict[str, Any],
) -> None:
    """
    Handles child span creation with Anthropic / Claude Code message format for Arize/Phoenix.
    """
    messages: List[Dict[str, Any]] = kwargs.get("messages", [])
    choices: List[Dict[str, Any]] = response_obj.get("choices", [])
    model_name: Optional[str] = kwargs.get("model")

    span_definitions: List[Tuple[str, Dict[str, Any], str]] = []

    # ENG2-1036: emit child spans only for the latest turn, not the full history.
    # Each of the three loops below previously walked all of `messages` from i=0,
    # producing O(L) child spans per call and O(L²) child spans per session.
    # Starting from the tail caps each loop at one or two iterations, keeping the
    # latest-turn detail Phoenix users want without re-emitting prior history.
    start_idx = max(0, len(messages) - 2)

    # Build internal prompt span data
    i = start_idx
    span_index = 0
    while i < len(messages):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        next_msg = messages[i + 1] if i + 1 < len(messages) else None

        span_name = f"Claude_Code_Internal_Prompt_{span_index}"

        if role == "user" and next_msg and next_msg.get("role") == "assistant":
            # paired messages (user & assistant)
            input_value, input_mime = detect_mimetype_and_serialize(content)
            output_value, output_mime = detect_mimetype_and_serialize(
                next_msg.get("content", "")
            )
            attrs = build_input_output_attributes(
                input_value, input_mime, output_value, output_mime, i, i + 1
            )
            span_definitions.append((span_name, attrs, OpenInferenceSpanKindValues.LLM.value))
            i += 2
        else:
            # single message (either user or assistant)
            content_value, content_mime = detect_mimetype_and_serialize(content)
            attrs = build_single_message_attributes(role, i, content_value, content_mime)
            span_definitions.append((span_name, attrs, OpenInferenceSpanKindValues.LLM.value))
            i += 1

        span_index += 1

    # Build internal tool span data
    i = start_idx
    span_index = 0
    while i < len(messages):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        next_msg = messages[i + 1] if i + 1 < len(messages) else None

        span_name = f"Claude_Code_Internal_Tool_{span_index}"

        if role == "assistant" and next_msg and next_msg.get("role") == "user":
            # paired messages (assistant & user)
            input_value, input_mime = detect_mimetype_and_serialize(content)
            output_value, output_mime = detect_mimetype_and_serialize(
                next_msg.get("content", "")
            )
            attrs = build_input_output_attributes(
                input_value, input_mime, output_value, output_mime, i, i + 1
            )
            span_definitions.append((span_name, attrs, OpenInferenceSpanKindValues.TOOL.value))
            i += 2
        else:
            i += 1

        span_index += 1

    # Build individual tool span data
    i = start_idx
    while i < len(messages):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        next_msg = messages[i + 1] if i + 1 < len(messages) else None

        if role == "assistant" and next_msg and next_msg.get("role") == "user":
            tool_use_value, tool_use_mime = detect_mimetype_and_serialize(content)
            tool_result_value, tool_result_mime = detect_mimetype_and_serialize(
                next_msg.get("content", "")
            )

            tool_use_json = try_parse_json(tool_use_value) if tool_use_mime == OpenInferenceMimeTypeValues.JSON.value else None
            tool_result_json = try_parse_json(tool_result_value) if tool_result_mime == OpenInferenceMimeTypeValues.JSON.value else None

            if isinstance(tool_use_json, list):
                for tool_use_obj in tool_use_json:
                    tool_use_type = tool_use_obj.get("type", "")
                    tool_use_id = tool_use_obj.get("id", "")
                    tool_use_name = tool_use_obj.get("name", "")
                    if tool_use_type == "tool_use":
                        input_value, input_mime = detect_mimetype_and_serialize(tool_use_obj)
                        output_value, output_mime = f"[ToolResultObject]", OpenInferenceMimeTypeValues.TEXT.value

                        if isinstance(tool_result_json, list):
                            for tool_result_obj in tool_result_json:
                                tool_result_type = tool_result_obj.get("type", "")
                                tool_result_id = tool_result_obj.get("tool_use_id", "")
                                tool_result_content = tool_result_obj.get("content", "")
                                if tool_result_type == "tool_result" and tool_result_id == tool_use_id:
                                    output_value, output_mime = tool_result_content, OpenInferenceMimeTypeValues.TEXT.value
                                    break

                        attrs = build_input_output_attributes(
                            input_value, input_mime, output_value, output_mime, i, i + 1
                        )
                        span_name = f"Claude_Code_Tool_{tool_use_name}"
                        attrs["claude_code_tool_name"] = tool_use_name
                        span_definitions.append((span_name, attrs, OpenInferenceSpanKindValues.TOOL.value))
            i += 2
        else:
            i += 1

    # Build final output span data
    for idx, choice in enumerate(choices):
        msg = choice.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        output_value, output_mime = detect_mimetype_and_serialize(content)

        span_name = f"Claude_Code_Final_Output_{idx}"
        attrs = {
            SpanAttributes.OUTPUT_VALUE: output_value,
            SpanAttributes.OUTPUT_MIME_TYPE: output_mime,
            "llm.message.role": role,
            "llm.message.index": idx,
        }
        span_definitions.append((span_name, attrs, OpenInferenceSpanKindValues.LLM.value))

    # Create spans in order
    for span_name, attrs, span_kind in reversed(span_definitions):
        create_anthropic_llm_child_span(tracer, parent_span, span_name, attrs, span_kind, model_name)


def set_attributes(span: Span, kwargs, response_obj):  # noqa: PLR0915
    """
    Populates span with OpenInference-compliant LLM attributes for Arize and Phoenix tracing.
    """
    try:
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_object not found in kwargs")

        #############################################
        ############ LLM CALL METADATA ##############
        #############################################

        # Set custom metadata for observability and trace enrichment.
        metadata = (
            standard_logging_payload.get("metadata")
            if standard_logging_payload
            else None
        )
        if metadata is not None:
            safe_set_attribute(span, SpanAttributes.METADATA, safe_dumps(metadata))

        #############################################
        ########## LLM Request Attributes ###########
        #############################################

        # The name of the LLM a request is being made to.
        if kwargs.get("model"):
            safe_set_attribute(
                span,
                SpanAttributes.LLM_MODEL_NAME,
                kwargs.get("model"),
            )

        # The LLM request type.
        safe_set_attribute(
            span,
            "llm.request.type",
            standard_logging_payload["call_type"],
        )

        # The Generative AI Provider: Azure, OpenAI, etc.
        safe_set_attribute(
            span,
            SpanAttributes.LLM_PROVIDER,
            litellm_params.get("custom_llm_provider", "Unknown"),
        )

        # The maximum number of tokens the LLM generates for a request.
        if optional_params.get("max_tokens"):
            safe_set_attribute(
                span,
                "llm.request.max_tokens",
                optional_params.get("max_tokens"),
            )

        # The temperature setting for the LLM request.
        if optional_params.get("temperature"):
            safe_set_attribute(
                span,
                "llm.request.temperature",
                optional_params.get("temperature"),
            )

        # The top_p sampling setting for the LLM request.
        if optional_params.get("top_p"):
            safe_set_attribute(
                span,
                "llm.request.top_p",
                optional_params.get("top_p"),
            )

        # Indicates whether response is streamed.
        safe_set_attribute(
            span,
            "llm.is_streaming",
            str(optional_params.get("stream", False)),
        )

        # Logs the user ID if present.
        if optional_params.get("user"):
            safe_set_attribute(
                span,
                "llm.user",
                optional_params.get("user"),
            )

        # The unique identifier for the completion.
        if response_obj and response_obj.get("id"):
            safe_set_attribute(span, "llm.response.id", response_obj.get("id"))

        # The model used to generate the response.
        if response_obj and response_obj.get("model"):
            safe_set_attribute(
                span,
                "llm.response.model",
                response_obj.get("model"),
            )

        # Required by OpenInference to mark span as LLM kind.
        safe_set_attribute(
            span,
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        messages = kwargs.get("messages")

        # for /chat/completions
        # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
        if messages:
            last_message = messages[-1]
            safe_set_attribute(
                span,
                SpanAttributes.INPUT_VALUE,
                last_message.get("content", ""),
            )

            # ENG2-1036: emit only the LAST message as an llm.input_messages.* attribute.
            # Previously we attached all N prior messages on every span, producing O(L²)
            # bytes per session. The most recent turn is sufficient for span-page detail;
            # full history reconstruction belongs in DAL post-ingest, not on every span.
            #
            # Use index 0 (not len(messages)-1) for the single emitted message:
            # OpenInference consumers expect contiguous 0-based indices when reading
            # arrays-as-prefixed-attributes. A non-zero sole index (e.g. ".10.role" with
            # no ".0.role" present) can be silently dropped or misaligned by some
            # consumers — Phoenix happens to normalize on read, but we shouldn't rely
            # on that. The "position in conversation" meaning is moot anyway since
            # we're emitting only one message.
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                last_message.get("role"),
            )
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                last_message.get("content", ""),
            )

        # Capture tools (function definitions) used in the LLM call.
        tools = optional_params.get("tools")
        if tools:
            for idx, tool in enumerate(tools):
                function = tool.get("function")
                if not function:
                    continue
                prefix = f"{SpanAttributes.LLM_TOOLS}.{idx}"
                safe_set_attribute(
                    span, f"{prefix}.{SpanAttributes.TOOL_NAME}", function.get("name")
                )
                safe_set_attribute(
                    span,
                    f"{prefix}.{SpanAttributes.TOOL_DESCRIPTION}",
                    function.get("description"),
                )
                safe_set_attribute(
                    span,
                    f"{prefix}.{SpanAttributes.TOOL_PARAMETERS}",
                    json.dumps(function.get("parameters")),
                )

        # Capture tool calls made during function-calling LLM flows.
        functions = optional_params.get("functions")
        if functions:
            for idx, function in enumerate(functions):
                prefix = f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{idx}"
                safe_set_attribute(
                    span,
                    f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    function.get("name"),
                )

        # Capture invocation parameters and user ID if available.
        model_params = (
            standard_logging_payload.get("model_parameters")
            if standard_logging_payload
            else None
        )
        if model_params:
            # ENG2-1476: strip tool schemas before dumping invocation_parameters.
            # Claude Code sessions attach the full MCP tools schema (~500KB) to
            # every call; it is identical across calls, so per-span storage is
            # pure duplication (same family as the ENG2-1036 message-history
            # dedup). Keep a count so "how many tools were attached" stays
            # observable; full schemas belong in a per-session store, not spans.
            slim_params = {
                k: v
                for k, v in model_params.items()
                if k not in ("tools", "functions")
            }
            for stripped_key in ("tools", "functions"):
                stripped = model_params.get(stripped_key)
                if isinstance(stripped, (list, tuple)) and stripped:
                    slim_params[f"{stripped_key}_count"] = len(stripped)
                    verbose_logger.debug(
                        "[arize] stripped %s from invocation_parameters: %d entries, ~%d bytes",
                        stripped_key,
                        len(stripped),
                        len(safe_dumps(stripped)),
                    )
            # The Generative AI Provider: Azure, OpenAI, etc.
            safe_set_attribute(
                span,
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                safe_dumps(slim_params),
            )

            if model_params.get("user"):
                user_id = model_params.get("user")
                if user_id is not None:
                    safe_set_attribute(span, SpanAttributes.USER_ID, user_id)

        #############################################
        ########## LLM Response Attributes ##########
        #############################################

        # Captures response tokens, message, and content.
        if hasattr(response_obj, "get"):
            for idx, choice in enumerate(response_obj.get("choices", [])):
                response_message = choice.get("message", {})
                safe_set_attribute(
                    span,
                    SpanAttributes.OUTPUT_VALUE,
                    response_message.get("content", ""),
                )

                # This shows up under `output_messages` tab on the span page.
                prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}"
                safe_set_attribute(
                    span,
                    f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                    response_message.get("role"),
                )
                safe_set_attribute(
                    span,
                    f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                    response_message.get("content", ""),
                )

            # Token usage info.
            usage = response_obj and response_obj.get("usage")
            if usage:
                safe_set_attribute(
                    span,
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    usage.get("total_tokens"),
                )

                # The number of tokens used in the LLM response (completion).
                safe_set_attribute(
                    span,
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    usage.get("completion_tokens"),
                )

                # The number of tokens used in the LLM prompt.
                safe_set_attribute(
                    span,
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    usage.get("prompt_tokens"),
                )

                # Response-side detail the semconv has slots for and this logger never
                # filled (Teraflop ENG2-1561): reasoning tokens, and the prompt cache
                # split. Without them a consumer reads only the three totals and has to
                # dig the rest out of metadata.usage_object, if it was logged at all.
                details = _usage_field(usage, "completion_tokens_details")
                reasoning = _usage_field(details, "reasoning_tokens")
                if reasoning is not None:
                    safe_set_attribute(
                        span,
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
                        reasoning,
                    )
                cache_read = _usage_field(usage, "cache_read_input_tokens")
                if cache_read is None:
                    cache_read = _usage_field(
                        _usage_field(usage, "prompt_tokens_details"), "cached_tokens"
                    )
                if cache_read is not None:
                    safe_set_attribute(
                        span,
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
                        cache_read,
                    )
                cache_write = _usage_field(usage, "cache_creation_input_tokens")
                if cache_write is not None:
                    safe_set_attribute(
                        span,
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
                        cache_write,
                    )

            # Why the model stopped. Anthropic's stop_reason arrives normalised as the
            # OpenAI finish_reason on the first choice; it was never written anywhere.
            choices = response_obj and response_obj.get("choices")
            if choices:
                finish_reason = _usage_field(choices[0], "finish_reason")
                if finish_reason is not None:
                    safe_set_attribute(span, "llm.response.finish_reason", finish_reason)

        #############################################
        ########## Anthropic / Claude Code ##########
        #############################################

        llm_request_type = standard_logging_payload.get("call_type", "Unknown")
        if llm_request_type == "anthropic_messages":
            tracer = trace.get_tracer(LITELLM_TRACER_NAME)
            handle_anthropic_claude_code_tracing(tracer, span, kwargs, response_obj)

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set OpenInference span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)
