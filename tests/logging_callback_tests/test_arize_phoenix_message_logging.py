"""ENG2-1461: `callback_settings: arize_phoenix: message_logging: false` must
actually reach the arize_phoenix logger and suppress the raw-request span.

The raw-request child span ("raw_gen_ai_request", emitted by
`OpenTelemetry.set_raw_request_attributes`) dumps the FULL request —
entire message history + tool definitions + system prompt — as
`llm.<provider>.*` attributes on EVERY call. For agent sessions that is
O(L²) storage (the ENG2-1036 bleed, resurfaced through this second code
path). The fix forwards proxy `callback_settings` into the arize_phoenix
logger init so `message_logging: false` gates that span off, while Span 1
(latest-turn content via set_arize_phoenix_attributes) is unaffected.

No network, no keys: drives `_handle_sucess` directly with synthetic kwargs
and asserts on an InMemorySpanExporter.
"""

import os
import sys
import types
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

RAW_SPAN_NAME = "raw_gen_ai_request"

# OTEL's global TracerProvider can only be set once per process; every
# OpenTelemetry() instance after the first keeps the first provider (and its
# exporter). Share one exporter so all instances' spans land somewhere we can
# read, and clear() it per run.
exporter = InMemorySpanExporter()

FAT_REQUEST = {
    "model": "claude-opus-4-7",
    "system": "You are a helpful assistant." * 50,
    "tools": [{"name": f"tool_{i}", "input_schema": {"type": "object"}} for i in range(40)],
    "messages": [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " * 200}
        for i in range(30)
    ],
}


def _synthetic_kwargs() -> dict:
    return {
        "model": "claude-opus-4-7",
        "call_type": "acompletion",
        "messages": FAT_REQUEST["messages"],
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "anthropic", "metadata": {}},
        "additional_args": {"complete_input_dict": FAT_REQUEST},
        "standard_logging_object": {},
    }


def _run_handler(message_logging: bool) -> list:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter.clear()
    otel = OpenTelemetry(
        config=OpenTelemetryConfig(exporter=exporter),
        callback_name="arize_phoenix",
        message_logging=message_logging,
    )
    # Bypass OTEL's set-once GLOBAL TracerProvider (another test in this
    # process may already own it): give this instance a private provider wired
    # to our exporter, so span capture is order-independent.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel.tracer = provider.get_tracer("arize-phoenix-message-logging-test")
    otel._handle_sucess(
        kwargs=_synthetic_kwargs(),
        response_obj=litellm.ModelResponse(),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    return exporter.get_finished_spans()


def test_message_logging_true_emits_raw_request_span():
    # Control: the default behavior emits the raw-request span with the full
    # provider payload — proving the test observes the thing we suppress.
    spans = _run_handler(message_logging=True)
    raw = [s for s in spans if s.name == RAW_SPAN_NAME]
    assert len(raw) == 1
    assert any(k.startswith("llm.anthropic.") for k in raw[0].attributes)


def test_message_logging_false_skips_raw_request_span():
    spans = _run_handler(message_logging=False)
    assert [s for s in spans if s.name == RAW_SPAN_NAME] == []
    # Span 1 (the litellm_request span) must still exist.
    assert len(spans) >= 1
    # And no span anywhere may carry the fat raw-request payload.
    for span in spans:
        assert not any(k.startswith("llm.anthropic.") for k in span.attributes)


def test_arize_phoenix_init_forwards_callback_settings(monkeypatch):
    # The proxy config path: callback_settings.arize_phoenix.message_logging
    # must reach the constructed logger (it was silently ignored before).
    from litellm.litellm_core_utils import litellm_logging as ll

    proxy_stub = types.ModuleType("litellm.proxy.proxy_server")
    proxy_stub.callback_settings = {"arize_phoenix": {"message_logging": False}}
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_stub)
    monkeypatch.setattr(ll, "_in_memory_loggers", [], raising=True)

    from litellm.integrations.arize.arize_phoenix import ArizePhoenixLogger
    from litellm.types.integrations.arize_phoenix import ArizePhoenixConfig

    monkeypatch.setattr(
        ArizePhoenixLogger,
        "get_arize_phoenix_config",
        staticmethod(
            lambda: ArizePhoenixConfig(
                otlp_auth_headers=None,
                protocol="otlp_http",
                endpoint="http://localhost:9999/v1/traces",
            )
        ),
    )

    logger = ll._init_custom_logger_compatible_class(
        logging_integration="arize_phoenix",
        internal_usage_cache=None,
        llm_router=None,
    )
    assert isinstance(logger, OpenTelemetry)
    assert logger.callback_name == "arize_phoenix"
    assert logger.message_logging is False
