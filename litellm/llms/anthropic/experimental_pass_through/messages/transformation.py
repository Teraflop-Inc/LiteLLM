from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams

from ...common_utils import AnthropicError

DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return [
            "messages",
            "model",
            "system",
            "max_tokens",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        import os
        import logging
        
        # High visibility debug print to stdout
        print(f"[OAUTH DEBUG] validate_anthropic_messages_environment CALLED for model: {model}")
        print(f"[OAUTH DEBUG] litellm_params keys: {list(litellm_params.keys())}")
        print(f"[OAUTH DEBUG] oauth_pass_through: {litellm_params.get('oauth_pass_through', 'NOT SET')}")
        print(f"[OAUTH DEBUG] headers keys: {list(headers.keys())}")
        print(f"[OAUTH DEBUG] headers content: {headers}")
        print(f"[OAUTH DEBUG] About to start OAuth token detection logic...")
        
        logger = logging.getLogger(__name__)
        logger.info(f"[OAuth Debug] validate_anthropic_messages_environment called for model: {model}")
        
        # OAuth authentication logic - check for OAuth token from auth layer
        oauth_token = None
        
        # NEW: Check if the api_key is an OAuth token (from user_api_key_auth)
        api_key_from_auth = litellm_params.get("api_key")
        print(f"[OAUTH DEBUG] api_key from litellm_params: {api_key_from_auth[:15] if api_key_from_auth else 'None'}...")
        
        # Check if we have proxy_server_request with auth data
        proxy_server_request = litellm_params.get("proxy_server_request")
        print(f"[OAUTH DEBUG] proxy_server_request type: {type(proxy_server_request)}")
        print(f"[OAUTH DEBUG] proxy_server_request content: {proxy_server_request}")
        if proxy_server_request and hasattr(proxy_server_request, 'api_key'):
            proxy_api_key = getattr(proxy_server_request, 'api_key', None)
            print(f"[OAUTH DEBUG] proxy_server_request.api_key: {proxy_api_key[:15] if proxy_api_key else 'None'}...")
            if proxy_api_key and proxy_api_key.startswith("sk-ant-oat"):
                oauth_token = proxy_api_key
                print(f"[OAUTH DEBUG] OAuth token detected from proxy_server_request.api_key: {oauth_token[:15]}...")
        elif proxy_server_request:
            # Check all attributes of proxy_server_request
            print(f"[OAUTH DEBUG] proxy_server_request attributes: {dir(proxy_server_request)}")
            for attr in dir(proxy_server_request):
                if not attr.startswith('_'):
                    value = getattr(proxy_server_request, attr)
                    print(f"[OAUTH DEBUG] proxy_server_request.{attr}: {value}")
                    if isinstance(value, str) and value.startswith("sk-ant-oat"):
                        oauth_token = value
                        print(f"[OAUTH DEBUG] Found OAuth token in {attr}: {oauth_token[:15]}...")
        
        if oauth_token is None and api_key_from_auth and api_key_from_auth.startswith("sk-ant-oat"):
            oauth_token = api_key_from_auth
            print(f"[OAUTH DEBUG] OAuth token detected from auth layer api_key: {oauth_token[:15]}...")
        
        # Fallback: Check if OAuth token was detected in the auth layer metadata
        if oauth_token is None and proxy_server_request and hasattr(proxy_server_request, 'metadata'):
            print(f"[OAUTH DEBUG] Found proxy_server_request metadata: {getattr(proxy_server_request, 'metadata', {})}")
            metadata = getattr(proxy_server_request, 'metadata', {})
            if metadata.get('oauth_pass_through') and metadata.get('oauth_token'):
                oauth_token = metadata['oauth_token']
                print(f"[OAUTH DEBUG] Using OAuth token from auth layer metadata: {oauth_token[:15]}...")
        
        # Fallback: Check headers for OAuth (legacy method, may not work due to header stripping)
        if oauth_token is None and litellm_params.get("oauth_pass_through", False):
            logger.info(f"[OAuth Debug] oauth_pass_through enabled, checking headers")
            logger.info(f"[OAuth Debug] Headers keys: {list(headers.keys())}")
            
            # Handle case-insensitive header lookup
            auth_header = headers.get("authorization") or headers.get("Authorization", "")
            logger.info(f"[OAuth Debug] Found auth header: {auth_header[:50] if auth_header else 'None'}...")
            
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "")
                # Detect OAuth token format (sk-ant-oat01-...)
                if token.startswith("sk-ant-oat"):
                    oauth_token = token
                    logger.info(f"[OAuth Debug] OAuth token detected: {oauth_token[:15]}...")
                else:
                    logger.info(f"[OAuth Debug] Bearer token is not OAuth format: {token[:15]}...")
            else:
                logger.info(f"[OAuth Debug] No valid Bearer token found")
        
        # Case 4: OAuth from file (with master key, LiteLLM manages OAuth)
        if oauth_token is None and litellm_params.get("oauth_token_file"):
            try:
                import json
                import time
                with open(litellm_params["oauth_token_file"], 'r') as f:
                    credentials = json.load(f)
                
                # Check if token is expired (expiresAt is in milliseconds)
                if credentials.get("expiresAt", 0) >= time.time() * 1000:
                    oauth_token = credentials.get("accessToken")
                    logger.info(f"[OAuth Debug] OAuth token loaded from file: {oauth_token[:15] if oauth_token else 'None'}...")
                else:
                    logger.info(f"[OAuth Debug] OAuth token from file is expired")
            except Exception as e:
                logger.info(f"[OAuth Debug] Failed to load OAuth token from file: {e}")

        # CRITICAL FIX: For OAuth pass-through, extract OAuth token from request metadata
        oauth_pass_through = litellm_params.get("oauth_pass_through", False)
        print(f"[OAUTH DEBUG] Condition check: oauth_pass_through={oauth_pass_through}, oauth_token is None={oauth_token is None}")
        
        if oauth_pass_through and oauth_token is None:
            print(f"[OAUTH DEBUG] OAuth pass-through enabled, attempting to extract OAuth token from request metadata")
            
            # Check litellm_metadata for oauth token (this is where user auth info goes)
            litellm_metadata = litellm_params.get("litellm_metadata", {})
            print(f"[OAUTH DEBUG] litellm_metadata keys: {list(litellm_metadata.keys())}")
            print(f"[OAUTH DEBUG] litellm_metadata content: {litellm_metadata}")
            
            # Also check if user_api_key_hash is present
            user_api_key_hash = litellm_metadata.get("user_api_key_hash")
            print(f"[OAUTH DEBUG] user_api_key_hash from metadata: {user_api_key_hash[:20] if user_api_key_hash else 'None'}...")
            
            # The OAuth token should be in the metadata from the auth layer
            user_api_key_hash = litellm_metadata.get("user_api_key_hash")
            if user_api_key_hash and user_api_key_hash.startswith("sk-ant-oat"):
                oauth_token = user_api_key_hash
                print(f"[OAUTH DEBUG] Found OAuth token in user_api_key_hash: {oauth_token[:20]}...")
            
            # Fallback: If no OAuth token found but pass-through is enabled, this means
            # we should use the actual client's authorization header token
            if oauth_token is None:
                print(f"[OAUTH DEBUG] No OAuth token in metadata, but pass-through enabled - extracting from original request")
                # In this case, we need to reconstruct the original OAuth token
                # Since headers are stripped, we'll check the request metadata for it
                user_api_key_metadata = litellm_metadata.get("user_api_key_metadata", {})
                if user_api_key_metadata.get("oauth_pass_through"):
                    # This indicates OAuth was detected but token wasn't preserved
                    # We need to extract it from the original request somehow
                    oauth_token = user_api_key_metadata.get("oauth_token", "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")
                    print(f"[OAUTH DEBUG] Using OAuth token from user metadata: {oauth_token[:20]}...")

        # Set authentication headers
        if oauth_token:
            logger.info(f"[OAuth Debug] Using OAuth authentication")
            print(f"[OAUTH DEBUG] Setting authorization header with OAuth token: {oauth_token[:20]}...")
            
            # Set the Bearer token for OAuth authentication
            headers["authorization"] = f"Bearer {oauth_token}"
                
            # Add OAuth beta headers for Claude Code integration
            if "anthropic-beta" not in headers:
                oauth_betas = ["oauth-2025-04-20", "claude-code-20250219", "interleaved-thinking-2025-05-14", "fine-grained-tool-streaming-2025-05-14"]
                headers["anthropic-beta"] = ",".join(oauth_betas)
                
            print(f"[OAUTH DEBUG] Final OAuth headers set: authorization=Bearer {oauth_token[:20]}..., anthropic-beta={headers.get('anthropic-beta', 'not set')}")
        else:
            logger.info(f"[OAuth Debug] Using API key authentication")
            # Fallback to API key authentication (existing logic)
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            if "x-api-key" not in headers and api_key:
                headers["x-api-key"] = api_key
        
        # Set standard headers
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        logger.info(f"[OAuth Debug] Final headers keys: {list(headers.keys())}")
        return headers, api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        No transformation is needed for Anthropic messages


        This takes in a request in the Anthropic /v1/messages API spec -> transforms it to /v1/messages API spec (i.e) no transformation is needed
        """
        max_tokens = anthropic_messages_optional_request_params.pop("max_tokens", None)
        if max_tokens is None:
            raise AnthropicError(
                message="max_tokens is required for Anthropic /v1/messages API",
                status_code=400,
            )
        ####### get required params for all anthropic messages requests ######
        anthropic_messages_request: AnthropicMessagesRequest = AnthropicMessagesRequest(
            messages=messages,
            max_tokens=max_tokens,
            model=model,
            **anthropic_messages_optional_request_params,
        )
        return dict(anthropic_messages_request)

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """
        No transformation is needed for Anthropic messages, since we want the response in the Anthropic /v1/messages API spec
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise AnthropicError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return AnthropicMessagesResponse(**raw_response_json)

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )

        # Use the shared streaming handler for Anthropic
        handler = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )
        return handler.get_async_streaming_response_iterator(
            httpx_response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )
