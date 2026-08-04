import json
import os
import sys
from typing import Optional

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio

import pytest

import litellm
from litellm.integrations._types.open_inference import (
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from litellm.integrations.arize.arize import ArizeLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Choices, StandardCallbackDynamicParams


def test_arize_set_attributes():
    """
    Test setting attributes for Arize, including all custom LLM attributes.
    Ensures that the correct span attributes are being added during a request.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import ModelResponse

    span = MagicMock()  # Mocked tracing span to test attribute setting

    # Construct kwargs to simulate a real LLM request scenario
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Basic Request Content"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {"key_1": "value_1", "key_2": None},
            "call_type": "completion",
        },
        "optional_params": {
            "max_tokens": "100",
            "temperature": "1",
            "top_p": "5",
            "stream": False,
            "user": "test_user",
            "tools": [
                {
                    "function": {
                        "name": "get_weather",
                        "description": "Fetches weather details.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["location"],
                        },
                    }
                }
            ],
            "functions": [{"name": "get_weather"}, {"name": "get_stock_price"}],
        },
        "litellm_params": {"custom_llm_provider": "openai"},
    }

    # Simulated LLM response object
    response_obj = ModelResponse(
        usage={"total_tokens": 100, "completion_tokens": 60, "prompt_tokens": 40},
        choices=[
            Choices(message={"role": "assistant", "content": "Basic Response Content"})
        ],
        model="gpt-4o",
        id="chatcmpl-ID",
    )

    # Apply attribute setting via ArizeLogger
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    # Validate that the expected number of attributes were set
    assert span.set_attribute.call_count == 28

    # Metadata attached to the span
    span.set_attribute.assert_any_call(
        SpanAttributes.METADATA, json.dumps({"key_1": "value_1", "key_2": None})
    )

    # Basic LLM information
    span.set_attribute.assert_any_call(SpanAttributes.LLM_MODEL_NAME, "gpt-4o")
    span.set_attribute.assert_any_call("llm.request.type", "completion")
    span.set_attribute.assert_any_call(SpanAttributes.LLM_PROVIDER, "openai")

    # LLM generation parameters
    span.set_attribute.assert_any_call("llm.request.max_tokens", "100")
    span.set_attribute.assert_any_call("llm.request.temperature", "1")
    span.set_attribute.assert_any_call("llm.request.top_p", "5")

    # Streaming and user info
    span.set_attribute.assert_any_call("llm.is_streaming", "False")
    span.set_attribute.assert_any_call("llm.user", "test_user")

    # Response metadata
    span.set_attribute.assert_any_call("llm.response.id", "chatcmpl-ID")
    span.set_attribute.assert_any_call("llm.response.model", "gpt-4o")
    span.set_attribute.assert_any_call(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")

    # Request message content and metadata
    span.set_attribute.assert_any_call(
        SpanAttributes.INPUT_VALUE, "Basic Request Content"
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
        "user",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
        "Basic Request Content",
    )

    # Tool call definitions and function names
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}", "get_weather"
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_DESCRIPTION}",
        "Fetches weather details.",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_PARAMETERS}",
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            }
        ),
    )

    # Tool calls captured from optional_params
    span.set_attribute.assert_any_call(
        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
        "get_weather",
    )
    span.set_attribute.assert_any_call(
        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
        "get_stock_price",
    )

    # Invocation parameters
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_INVOCATION_PARAMETERS, '{"user": "test_user"}'
    )

    # User ID
    span.set_attribute.assert_any_call(SpanAttributes.USER_ID, "test_user")

    # Output message content
    span.set_attribute.assert_any_call(
        SpanAttributes.OUTPUT_VALUE, "Basic Response Content"
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
        "assistant",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
        "Basic Response Content",
    )

    # Token counts
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 100)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 60)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 40)


def test_arize_invocation_parameters_strips_tool_schemas():
    """
    ENG2-1476: model_parameters carrying a fat tools schema (the Claude Code /
    MCP case, ~500KB identical on every call) must NOT be dumped into
    llm.invocation_parameters. The tools/functions keys are stripped and
    replaced with counts; all other invocation params are preserved.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import ModelResponse

    span = MagicMock()

    # Anthropic-format tools (top-level name/input_schema, no "function"
    # wrapper), padded to be measurably large like a real MCP tools blob.
    fat_tools = [
        {
            "name": f"tool_{i}",
            "description": "x" * 2000,
            "input_schema": {
                "type": "object",
                "properties": {f"param_{j}": {"type": "string", "description": "y" * 200} for j in range(10)},
            },
        }
        for i in range(50)
    ]
    assert len(json.dumps(fat_tools)) > 100_000  # test-fixture sanity

    kwargs = {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "hi"}],
        "standard_logging_object": {
            "model_parameters": {
                "max_tokens": 64000,
                "stream": True,
                "thinking": {"type": "adaptive"},
                "user": "test_user",
                "tools": fat_tools,
            },
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 10, "completion_tokens": 5, "prompt_tokens": 5},
        choices=[Choices(message={"role": "assistant", "content": "hello"})],
        model="claude-opus-4-8",
        id="chatcmpl-ID",
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    invocation_calls = [
        c.args[1]
        for c in span.set_attribute.call_args_list
        if c.args[0] == SpanAttributes.LLM_INVOCATION_PARAMETERS
    ]
    assert len(invocation_calls) == 1
    emitted = json.loads(invocation_calls[0])

    # Tool schemas are gone, replaced by a count; everything else survives.
    assert "tools" not in emitted
    assert emitted["tools_count"] == 50
    assert emitted["max_tokens"] == 64000
    assert emitted["stream"] is True
    assert emitted["thinking"] == {"type": "adaptive"}
    assert emitted["user"] == "test_user"

    # The whole emitted blob is small — this is the acceptance criterion.
    assert len(invocation_calls[0]) < 1000

    # User ID extraction still works off the original params.
    span.set_attribute.assert_any_call(SpanAttributes.USER_ID, "test_user")


class TestArizeLogger(CustomLogger):
    """
    Custom logger implementation to capture standard_callback_dynamic_params.
    Used to verify that dynamic config keys are being passed to callbacks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_callback_dynamic_params: Optional[
            StandardCallbackDynamicParams
        ] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Capture dynamic params and print them for verification
        print("logged kwargs", json.dumps(kwargs, indent=4, default=str))
        self.standard_callback_dynamic_params = kwargs.get(
            "standard_callback_dynamic_params"
        )


@pytest.mark.asyncio
async def test_arize_dynamic_params():
    """
    Test to ensure that dynamic Arize keys (API key and space key)
    are received inside the callback logger at runtime.
    """
    test_arize_logger = TestArizeLogger()
    litellm.callbacks = [test_arize_logger]

    # Perform a mocked async completion call to trigger logging
    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Basic Request Content"}],
        mock_response="test",
        arize_api_key="test_api_key_dynamic",
        arize_space_key="test_space_key_dynamic",
    )

    # Allow for async propagation
    await asyncio.sleep(2)

    # Assert dynamic parameters were received in the callback
    assert test_arize_logger.standard_callback_dynamic_params is not None
    assert (
        test_arize_logger.standard_callback_dynamic_params.get("arize_api_key")
        == "test_api_key_dynamic"
    )
    assert (
        test_arize_logger.standard_callback_dynamic_params.get("arize_space_key")
        == "test_space_key_dynamic"
    )


def test_construct_dynamic_arize_headers():
    """
    Test the construct_dynamic_arize_headers method with various input scenarios.
    Ensures that dynamic Arize headers are properly constructed from callback parameters.
    """
    from litellm.types.utils import StandardCallbackDynamicParams

    # Test with all parameters present
    dynamic_params_full = StandardCallbackDynamicParams(
        arize_api_key="test_api_key", 
        arize_space_id="test_space_id"
    )
    arize_logger = ArizeLogger()
    
    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_full)
    expected_headers = {
        "api_key": "test_api_key",
        "arize-space-id": "test_space_id"
    }
    assert headers == expected_headers
        
    # Test with only space_id
    dynamic_params_space_id_only = StandardCallbackDynamicParams(
        arize_space_id="test_space_id"
    )
    
    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_space_id_only)
    expected_headers = {
        "arize-space-id": "test_space_id"
    }
    assert headers == expected_headers
    
    # Test with empty parameters dict
    dynamic_params_empty = StandardCallbackDynamicParams()
    
    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_empty)
    assert headers == {}

    # test with space key and api key
    dynamic_params_space_key_and_api_key = StandardCallbackDynamicParams(
        arize_space_key="test_space_key",
        arize_api_key="test_api_key"
    )
    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_space_key_and_api_key)
    expected_headers = {
        "arize-space-id": "test_space_key",
        "api_key": "test_api_key"
    }
