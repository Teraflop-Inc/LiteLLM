"""A request that passes the proxy key and carries its own Anthropic API key bills that key."""

import logging
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

CALLER_KEY = "sk-ant-api03-caller-test-key"


def _request(headers: dict, path: str = "/v1/messages") -> Request:
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.path = path
    request.url.__str__.return_value = f"http://localhost{path}"
    request.method = "POST"
    request.query_params = {}
    request.headers = {"content-type": "application/json", **headers}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


async def _pre_call(headers: dict, metadata: dict, path: str = "/v1/messages") -> dict:
    return await add_litellm_data_to_request(
        data={"model": "claude-haiku-5-5"},
        request=_request(headers, path),
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-proxy-key", metadata=metadata),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )


@pytest.mark.asyncio
async def test_caller_anthropic_key_is_carried_to_the_router():
    data = await _pre_call({"x-api-key": CALLER_KEY}, metadata={})
    assert data["litellm_params"]["caller_api_key"] == CALLER_KEY


@pytest.mark.asyncio
async def test_caller_key_is_only_honored_on_the_messages_route():
    data = await _pre_call(
        {"x-api-key": CALLER_KEY}, metadata={}, path="/v1/chat/completions"
    )
    assert "caller_api_key" not in (data.get("litellm_params") or {})


@pytest.mark.asyncio
async def test_non_anthropic_x_api_key_is_ignored():
    data = await _pre_call({"x-api-key": "not-an-anthropic-key"}, metadata={})
    assert "caller_api_key" not in (data.get("litellm_params") or {})


@pytest.mark.asyncio
async def test_oauth_pass_through_wins_over_a_caller_key():
    data = await _pre_call(
        {"x-api-key": CALLER_KEY},
        metadata={"oauth_pass_through": True, "oauth_token": "sk-ant-oat01-test"},
    )
    assert data["litellm_params"]["oauth_token"] == "sk-ant-oat01-test"
    assert "caller_api_key" not in data["litellm_params"]


@pytest.mark.asyncio
async def test_caller_and_proxy_keys_stay_out_of_stored_metadata():
    data = await _pre_call(
        {"x-api-key": CALLER_KEY, "x-litellm-api-key": "sk-proxy-gate-key"}, metadata={}
    )
    stored = str(data["litellm_metadata"]["headers"])
    assert CALLER_KEY not in stored
    assert "sk-proxy-gate-key" not in stored


@pytest.mark.asyncio
async def test_caller_key_never_reaches_an_info_log_line(caplog):
    # sf-litellm runs at INFO. Upstream's own DEBUG lines print raw request headers.
    with caplog.at_level(logging.INFO, logger=verbose_proxy_logger.name):
        await _pre_call({"x-api-key": CALLER_KEY}, metadata={})
    assert all(CALLER_KEY not in r.getMessage() for r in caplog.records)


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "claude-haiku-*",
                "litellm_params": {
                    "model": "claude-haiku-*",
                    "api_key": "sk-ant-api03-proxy-own-key",
                    "custom_llm_provider": "anthropic",
                },
            },
            {
                "model_name": "anthropic/*",
                "litellm_params": {
                    "model": "anthropic/*",
                    "api_key": "sk-ant-api03-proxy-own-key",
                },
            },
            {
                "model_name": "gpt-test",
                "litellm_params": {
                    "model": "openai/gpt-test",
                    "api_key": "sk-openai-own-key",
                },
            },
        ]
    )


async def _router_call(model: str = "claude-haiku-5-5", **kwargs) -> dict:
    seen: dict = {}

    async def upstream(**call_kwargs):
        seen.update(call_kwargs)
        return {"ok": True}

    await _router()._ageneric_api_call_with_fallbacks_helper(
        model=model, original_generic_function=upstream, **kwargs
    )
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-haiku-5-5", "anthropic/claude-haiku-5-5"])
async def test_router_uses_the_caller_key_instead_of_the_deployment_key(model):
    seen = await _router_call(model, litellm_params={"caller_api_key": CALLER_KEY})
    assert seen["api_key"] == CALLER_KEY


@pytest.mark.asyncio
async def test_router_never_hands_the_caller_key_to_another_provider():
    seen = await _router_call(
        "gpt-test", litellm_params={"caller_api_key": CALLER_KEY}
    )
    assert seen["api_key"] == "sk-openai-own-key"


@pytest.mark.asyncio
async def test_router_keeps_the_deployment_key_without_a_caller_key():
    seen = await _router_call()
    assert seen["api_key"] == "sk-ant-api03-proxy-own-key"
