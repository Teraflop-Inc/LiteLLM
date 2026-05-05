"""
ENG2-1036 — measure O(L²) trace emit growth in the Arize integration.

Simulates a Claude Code session of length L by calling
`litellm.integrations.arize._utils.set_attributes` L times with progressively
longer message histories. Captures every emitted span/attribute via an
in-memory exporter and reports cumulative counts per session length.

Usage:
    uv run python scripts/measure_l2_baseline.py
    # or:
    python scripts/measure_l2_baseline.py

Reports:
    - parent span count          (should be == L)
    - input_messages.* attrs     (sum should be L*(L+1)/2  →  O(L²))
    - child span count           (sum should be ~ L*L      →  O(L²))
    - child/L and child/L² ratios (the latter should be ~constant if O(L²))

Re-run after the dedup patch and compare. Same scenario, same numbers post-fix
should be O(L) instead of O(L²).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from litellm.integrations.arize import _utils


class ListExporter(SpanExporter):
    """Captures every finished span in memory for inspection."""

    def __init__(self) -> None:
        self.spans: List[Any] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans = []


_EXPORTER = ListExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)
_TRACER = trace.get_tracer("litellm")


def build_messages(num_pairs: int) -> List[Dict[str, Any]]:
    """Build a Claude Code-style messages array.

    Each pair is one user turn + one assistant turn that calls a tool.
    The final turn is a user `tool_result`, mirroring how Claude Code calls
    Anthropic mid-conversation.
    """
    msgs: List[Dict[str, Any]] = []
    for i in range(num_pairs):
        msgs.append({"role": "user", "content": f"please run command number {i}"})
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{i:04d}",
                        "name": "Bash",
                        "input": {"command": f"echo {i}"},
                    }
                ],
            }
        )
    if num_pairs > 0:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{num_pairs - 1:04d}",
                        "content": f"output of command {num_pairs - 1}",
                    }
                ],
            }
        )
    return msgs


def build_kwargs(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "messages": messages,
        "model": "claude-sonnet-4",
        "optional_params": {"max_tokens": 4096},
        "litellm_params": {"custom_llm_provider": "anthropic"},
        "standard_logging_object": {
            "call_type": "anthropic_messages",
            "metadata": {"session_id": "l2-baseline-test"},
            "model_parameters": {},
        },
    }


def build_response(call_idx: int) -> Dict[str, Any]:
    return {
        "id": f"resp_{call_idx:04d}",
        "model": "claude-sonnet-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"response for call {call_idx}",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100 + call_idx * 50,
            "completion_tokens": 20,
            "total_tokens": 120 + call_idx * 50,
        },
    }


def simulate_session(num_calls: int) -> Dict[str, int]:
    """Simulate `num_calls` LLM calls.

    Call K (1-indexed) sees K conversation pairs in its messages array,
    matching how a real Claude Code session grows turn over turn.
    """
    _EXPORTER.clear()

    for k in range(1, num_calls + 1):
        messages = build_messages(num_pairs=k)
        kwargs = build_kwargs(messages)
        response = build_response(k)

        with _TRACER.start_as_current_span(f"llm_call_{k:04d}") as parent_span:
            _utils.set_attributes(parent_span, kwargs, response)

    spans = list(_EXPORTER.spans)

    parent_spans = [s for s in spans if s.name.startswith("llm_call_")]
    child_spans = [s for s in spans if not s.name.startswith("llm_call_")]

    input_msg_attrs = 0
    for s in parent_spans:
        for key in s.attributes.keys():
            if key.startswith("llm.input_messages."):
                input_msg_attrs += 1

    by_kind: Dict[str, int] = {}
    for s in child_spans:
        prefix = s.name.rsplit("_", 1)[0] if "_" in s.name else s.name
        by_kind[prefix] = by_kind.get(prefix, 0) + 1

    return {
        "L": num_calls,
        "parent_spans": len(parent_spans),
        "child_spans": len(child_spans),
        "input_msg_attrs": input_msg_attrs,
        "by_kind": by_kind,
    }


def fmt_ratio(num: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{num / denom:>6.2f}"


def main() -> int:
    print("ENG2-1036 — L² baseline measurement\n")
    print(
        f"{'L':>4} | {'parent':>7} | {'child':>7} | {'in_msg_attrs':>13} "
        f"| {'child/L':>8} | {'child/L²':>9}"
    )
    print("-" * 66)

    rows = []
    for L in [1, 5, 10, 20, 30]:
        r = simulate_session(L)
        rows.append(r)
        print(
            f"{r['L']:>4} | {r['parent_spans']:>7} | {r['child_spans']:>7} "
            f"| {r['input_msg_attrs']:>13} | {fmt_ratio(r['child_spans'], L):>8} "
            f"| {fmt_ratio(r['child_spans'], L * L):>9}"
        )

    print("\nChild span breakdown for L=30 (last row):")
    for kind, count in sorted(rows[-1]["by_kind"].items(), key=lambda x: -x[1]):
        print(f"  {kind:<40} {count:>5}")

    print(
        "\nInterpretation:\n"
        "  - If child/L grows with L  → super-linear (the L² problem).\n"
        "  - If child/L² is roughly constant → confirmed quadratic.\n"
        "  - in_msg_attrs growing as L*(L+1)/2 confirms Source #1 "
        "(set_attributes parent-attr bloat).\n"
        "  - child_spans growing as ~L² confirms Source #2 "
        "(handle_anthropic_claude_code_tracing per-call child-span emit).\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
