"""Codex ChatGPT pass-through auth names the caller from a configured header (ENG2-402).

The ChatGPT OAuth JWT is forwarded unchecked on the Codex route, so every Codex span used to
carry the same user_id and no end user. With general_settings.user_header_name set, the
header the client sends becomes the end user, which pass-through logging writes to the span
as user_api_key_end_user_id.
"""

import pytest
from starlette.requests import Request

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

JWT = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0.c2ln"


def _request(headers: dict) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/teraflop-codex/responses",
        "raw_path": b"/teraflop-codex/responses",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


async def _auth(headers: dict):
    return await _user_api_key_auth_builder(
        request=_request(headers),
        api_key=f"Bearer {JWT}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        request_data={},
    )


@pytest.mark.asyncio
async def test_configured_header_becomes_end_user(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {"user_header_name": "x-dal-user"})
    auth = await _auth({"authorization": f"Bearer {JWT}", "x-dal-user": "dev@example.com"})
    assert auth.end_user_id == "dev@example.com"
    assert auth.user_id == "chatgpt-oauth-user"
    assert auth.metadata == {"oauth_pass_through": True, "harness": "codex"}


@pytest.mark.asyncio
async def test_no_end_user_without_config(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {})
    auth = await _auth({"authorization": f"Bearer {JWT}", "x-dal-user": "dev@example.com"})
    assert auth.end_user_id is None


@pytest.mark.asyncio
async def test_no_end_user_when_header_missing(monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", {"user_header_name": "x-dal-user"})
    auth = await _auth({"authorization": f"Bearer {JWT}"})
    assert auth.end_user_id is None


def test_the_identity_header_is_not_forwarded_upstream(monkeypatch):
    """The header names the caller for the proxy's logs; chatgpt.com never sees it."""
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        HttpPassThroughEndpointHelpers,
    )

    monkeypatch.setattr(proxy_server, "general_settings", {"user_header_name": "x-dal-user"})
    out = HttpPassThroughEndpointHelpers.drop_caller_identity_header(
        {"authorization": "Bearer t", "X-DAL-User": "dev@example.com", "x-other": "1"}
    )
    assert out == {"authorization": "Bearer t", "x-other": "1"}


def test_headers_untouched_without_the_setting(monkeypatch):
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        HttpPassThroughEndpointHelpers,
    )

    monkeypatch.setattr(proxy_server, "general_settings", {})
    headers = {"x-dal-user": "dev@example.com"}
    assert HttpPassThroughEndpointHelpers.drop_caller_identity_header(headers) == headers
