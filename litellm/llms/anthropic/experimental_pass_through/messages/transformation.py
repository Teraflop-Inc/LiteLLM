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
        
        logger = logging.getLogger(__name__)
        logger.info(f"[OAuth Debug] validate_anthropic_messages_environment called for model: {model}")
        
        # SIMPLIFIED OAuth pass-through logic
        oauth_token = None
        
        print(f"[OAUTH DEBUG] Checking OAuth pass-through for model: {model}")
        print(f"[OAUTH DEBUG] oauth_pass_through flag: {litellm_params.get('oauth_pass_through', False)}")
        print(f"[OAUTH DEBUG] api_key param: {api_key[:20] if api_key else 'None'}...")
        
        # Check if OAuth pass-through is enabled and we have a token from the auth layer
        if litellm_params.get("oauth_pass_through", False):
            print(f"[OAUTH DEBUG] OAuth pass-through enabled for model: {model}")
            
            # Primary method: Check if api_key parameter already contains the OAuth token
            # (passed from litellm_pre_call_utils.py)
            if api_key and api_key.startswith("sk-ant-oat"):
                oauth_token = api_key
                print(f"[OAUTH DEBUG] Using OAuth token from api_key parameter: {oauth_token[:20]}...")
            else:
                # Fallback: Check litellm_params for the token
                # First check for oauth_token (from router fix)
                oauth_token_from_params = litellm_params.get("oauth_token")
                api_key_from_params = litellm_params.get("api_key")
                
                if oauth_token_from_params:
                    oauth_token = oauth_token_from_params
                    print(f"[OAUTH DEBUG] Using OAuth token from litellm_params.oauth_token: {oauth_token[:20]}...")
                elif api_key_from_params and api_key_from_params.startswith("sk-ant-oat"):
                    oauth_token = api_key_from_params
                    print(f"[OAUTH DEBUG] Using OAuth token from litellm_params.api_key: {oauth_token[:20]}...")
                else:
                    print(f"[OAUTH DEBUG] WARNING: OAuth pass-through enabled but no OAuth token found!")
                    print(f"[OAUTH DEBUG] api_key: {api_key[:20] if api_key else 'None'}...")
                    print(f"[OAUTH DEBUG] litellm_params.api_key: {api_key_from_params[:20] if api_key_from_params else 'None'}...")
                    print(f"[OAUTH DEBUG] litellm_params.oauth_token: {oauth_token_from_params[:20] if oauth_token_from_params else 'None'}...")

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
