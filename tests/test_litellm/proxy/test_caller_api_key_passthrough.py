"""A caller's own Anthropic API key bills that key (decision 0014).

Sent alone, it passes through like a Claude login token, gated by the network. Sent beside a
LiteLLM key, the request takes the normal checks and the key is still forwarded.
"""

import logging
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from starlette.requests import Request as StarletteRequest

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import (
    _caller_key_pass_through,
    _user_api_key_auth_builder,
)
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

CALLER_KEY = "sk-ant-api03-caller-test-key"


def _starlette_request(headers: dict, path: str = "/v1/messages") -> StarletteRequest:
    return StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


@pytest.mark.parametrize(
    "headers, path, expected",
    [
        ({"x-api-key": CALLER_KEY}, "/v1/messages", CALLER_KEY),
        ({"x-api-key": CALLER_KEY, "x-litellm-api-key": "sk-proxy"}, "/v1/messages", None),
        ({"x-api-key": CALLER_KEY, "authorization": "Bearer sk-proxy"}, "/v1/messages", None),
        ({"x-api-key": CALLER_KEY}, "/v1/chat/completions", None),
        ({"x-api-key": "not-an-anthropic-key"}, "/v1/messages", None),
        ({"x-api-key": "sk-ant-svc01-service-account-key"}, "/v1/messages", "sk-ant-svc01-service-account-key"),
        ({"x-api-key": "sk-ant-usr01-personal-key"}, "/v1/messages", "sk-ant-usr01-personal-key"),
        ({"x-api-key": "sk-ant-oat01-login-token"}, "/v1/messages", None),
        ({"x-api-key": "sk-ant-admin01-admin-key"}, "/v1/messages", None),
    ],
)
def test_caller_key_passes_through_only_as_the_sole_credential(headers, path, expected):
    request = _starlette_request(headers, path)
    assert _caller_key_pass_through(route=path, request=request) == expected


@pytest.mark.asyncio
async def test_caller_key_alone_authenticates_as_pass_through(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {"user_header_name": "x-dal-user"})
    auth = await _user_api_key_auth_builder(
        request=_starlette_request({"x-api-key": CALLER_KEY, "x-dal-user": "dev@example.com"}),
        api_key=CALLER_KEY,
        azure_api_key_header="",
        anthropic_api_key_header=CALLER_KEY,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        request_data={},
    )
    assert auth.metadata == {"caller_key_pass_through": True}
    assert auth.user_id == "caller-key-user"
    assert auth.end_user_id == "dev@example.com"
    assert CALLER_KEY not in auth.model_dump_json(), "raw key kept on the auth object"


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
@pytest.mark.parametrize(
    "key",
    [CALLER_KEY, "sk-ant-svc01-service-account-key", "sk-ant-usr01-personal-key"],
)
async def test_caller_anthropic_key_is_carried_to_the_router(key):
    data = await _pre_call({"x-api-key": key}, metadata={})
    assert data["litellm_params"]["caller_api_key"] == key


@pytest.mark.asyncio
async def test_pass_through_flag_reaches_the_router():
    data = await _pre_call(
        {"x-api-key": CALLER_KEY}, metadata={"caller_key_pass_through": True}
    )
    assert data["litellm_params"]["caller_key_pass_through"] is True


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
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_caller_key_never_reaches_a_log_line(caplog, level):
    # sf-litellm runs at INFO, but DEBUG is what someone turns on to troubleshoot, and
    # upstream's DEBUG lines printed raw request headers and the whole request data.
    with caplog.at_level(level, logger=verbose_proxy_logger.name):
        await _pre_call({"x-api-key": CALLER_KEY}, metadata={})
    leaked = [r.getMessage()[:80] for r in caplog.records if CALLER_KEY in r.getMessage()]
    assert not leaked, leaked


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
async def test_router_uses_a_pass_through_caller_key_on_anthropic():
    seen = await _router_call(
        litellm_params={"caller_api_key": CALLER_KEY, "caller_key_pass_through": True}
    )
    assert seen["api_key"] == CALLER_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "litellm_params",
    [
        {"caller_api_key": CALLER_KEY, "caller_key_pass_through": True},
        {"oauth_pass_through": True, "oauth_token": "sk-ant-oat01-test"},
    ],
)
async def test_pass_through_credentials_never_spend_another_providers_key(litellm_params):
    with pytest.raises(litellm.AuthenticationError):
        await _router_call("gpt-test", litellm_params=litellm_params)


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


# --- Hardening (2026-10-20 review): the pass-through identities are granted without the
# credential being checked, so their routes must be exact and the request body must still
# pass the proxy's safety checks. Each case below reached the proxy's own deployment key
# (or an admin route) before.


@pytest.mark.parametrize(
    "path",
    [
        "/anthropic/v1/messages",  # pass-through route that swaps in the proxy's own key
        "/anthropic/v1/messages/batches",
        "/v1/messages/count_tokens",  # counts on the deployment's key, not the caller's
        "/v1/messages/batches",
        "/credentials/v1/messages",  # admin route with a catch-all path segment
        "/key/v1/messages/regenerate",
    ],
)
def test_caller_key_is_honored_on_v1_messages_only(path):
    request = _starlette_request({"x-api-key": CALLER_KEY}, path)
    assert _caller_key_pass_through(route=path, request=request) is None


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/v1/messages", True),
        ("/v1/messages/count_tokens", True),
        ("/anthropic/v1/messages", False),
        ("/credentials/v1/messages", False),
        ("/v1/messages/batches", False),
    ],
)
def test_claude_login_pass_through_routes_are_exact(path, expected):
    from litellm.proxy.auth.user_api_key_auth import _is_claude_login_route

    assert _is_claude_login_route(path) is expected


@pytest.mark.asyncio
async def test_caller_key_identity_is_not_proxy_admin(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {})
    auth = await _user_api_key_auth_builder(
        request=_starlette_request({"x-api-key": CALLER_KEY}),
        api_key=CALLER_KEY,
        azure_api_key_header="",
        anthropic_api_key_header=CALLER_KEY,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        request_data={},
    )
    assert auth.metadata == {"caller_key_pass_through": True}
    assert auth.user_role != "proxy_admin", "an unchecked credential got admin"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["api_base", "base_url"])
async def test_caller_key_request_body_cannot_redirect_the_call(monkeypatch, field):
    monkeypatch.setattr(proxy_server, "general_settings", {})
    with pytest.raises(Exception, match="not allowed in request body"):
        await _user_api_key_auth_builder(
            request=_starlette_request({"x-api-key": CALLER_KEY}),
            api_key=CALLER_KEY,
            azure_api_key_header="",
            anthropic_api_key_header=CALLER_KEY,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "claude-haiku-5-5", field: "https://evil.example"},
        )


@pytest.mark.asyncio
async def test_claude_login_request_body_cannot_redirect_the_call(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {})
    with pytest.raises(Exception, match="not allowed in request body"):
        await _user_api_key_auth_builder(
            request=_starlette_request({"authorization": "Bearer sk-ant-oat01-login"}),
            api_key="sk-ant-oat01-login",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "claude-haiku-5-5", "api_base": "https://evil.example"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "smuggled",
    [
        {"oauth_pass_through": True},
        {"oauth_pass_through": True, "oauth_token": "sk-ant-oat01-forged"},
        {"caller_key_pass_through": True},
        {"caller_api_key": "sk-ant-api03-someone-else"},
    ],
)
async def test_body_cannot_set_the_pass_through_markers(smuggled):
    data = await add_litellm_data_to_request(
        data={"model": "claude-haiku-5-5", "litellm_params": dict(smuggled)},
        request=_request({"x-api-key": CALLER_KEY}),
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed", metadata={}),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )
    params = data.get("litellm_params") or {}
    assert params.get("caller_api_key") == CALLER_KEY, "the real caller key still applies"
    assert not params.get("oauth_pass_through")
    assert "oauth_token" not in params
    assert not params.get("caller_key_pass_through"), "only the auth result sets this"


# --- Greptile on #21: count_tokens ran on the deployment's key for an unchecked login
# token, and login tokens still reached logs in slices.

LOGIN_TOKEN = "sk-ant-oat01-LOGINSECRETabcdefghijklmnopqrstuvwxyz0123456789"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata, expect_endpoint",
    [
        ({"oauth_pass_through": True, "oauth_token": LOGIN_TOKEN}, False),
        ({"caller_key_pass_through": True}, False),
        ({}, True),  # a LiteLLM key holder may still use the provider's counting endpoint
    ],
)
async def test_count_tokens_never_spends_the_deployment_key_for_a_pass_through_identity(
    monkeypatch, metadata, expect_endpoint
):
    from litellm.proxy.anthropic_endpoints import endpoints

    seen = {}

    async def fake_counter(request, call_endpoint=False):
        seen["call_endpoint"] = call_endpoint
        return {"total_tokens": 7}

    async def fake_body(request):
        return {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}]}

    monkeypatch.setattr(proxy_server, "token_counter", fake_counter)
    monkeypatch.setattr(endpoints, "_read_request_body", fake_body)
    out = await endpoints.count_tokens(
        request=_starlette_request({}, "/v1/messages/count_tokens"),
        user_api_key_dict=UserAPIKeyAuth(api_key="h", metadata=metadata),
    )
    assert out == {"input_tokens": 7}
    assert seen["call_endpoint"] is expect_endpoint


@pytest.mark.asyncio
async def test_login_token_never_reaches_a_log_line(caplog):
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    with caplog.at_level(logging.DEBUG):
        await _pre_call(
            {"authorization": f"Bearer {LOGIN_TOKEN}"},
            metadata={"oauth_pass_through": True, "oauth_token": LOGIN_TOKEN},
        )
        AnthropicModelInfo.detect_oauth_token(f"Bearer {LOGIN_TOKEN}")
        AnthropicModelInfo.extract_oauth_from_request(
            {"authorization": f"Bearer {LOGIN_TOKEN}"}, {"oauth_pass_through": True}
        )
    # Not even a slice: anything after the public prefix is the secret.
    import re

    leaked = [
        r.getMessage()[:80] for r in caplog.records if re.search(r"sk-ant-oat01-\w", r.getMessage())
    ]
    assert not leaked, leaked
