"""Claude Code tool spans when user messages follow the tool results (ENG2-1765)."""
import importlib.util
import json
import os

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "litellm", "integrations", "arize", "_utils.py")
spec = importlib.util.spec_from_file_location("fork_arize_utils", SRC)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def _spans(messages):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("parent") as parent:
        utils.handle_anthropic_claude_code_tracing(
            tracer, parent, {"messages": messages, "model": "m"},
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]})
    return [s for s in exporter.get_finished_spans() if s.name != "parent"]


TOOL_USE = {"role": "assistant", "content": [
    {"type": "text", "text": "reading"},
    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.py"}},
    {"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {"command": "ls"}}]}
RESULTS = {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file a"},
    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "a.py"}]}
REMINDER = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>be brief</system-reminder>",
                                          "cache_control": {"type": "ephemeral"}}]}


def _tools(spans):
    return {s.name: (json.loads(s.attributes["input.value"])["id"], s.attributes["output.value"])
            for s in spans if s.name.startswith("Claude_Code_Tool_")}


def test_tool_spans_when_the_results_end_the_request():
    spans = _spans([{"role": "user", "content": "go"}, TOOL_USE, RESULTS])
    assert _tools(spans) == {"Claude_Code_Tool_Read": ("toolu_1", "file a"),
                             "Claude_Code_Tool_Bash": ("toolu_2", "a.py")}


def test_tool_spans_when_reminders_follow_the_results():
    spans = _spans([{"role": "user", "content": "go"}, TOOL_USE, RESULTS, REMINDER])
    assert _tools(spans) == {"Claude_Code_Tool_Read": ("toolu_1", "file a"),
                             "Claude_Code_Tool_Bash": ("toolu_2", "a.py")}
    # The trailing reminder still gets its own prompt span, whole.
    prompts = [s for s in spans if s.name.startswith("Claude_Code_Internal_Prompt_")]
    assert any("be brief" in str(s.attributes.get("input.value", "")) for s in prompts)


def test_two_trailing_user_messages_still_emit_only_the_latest_turn():
    history = [{"role": "user", "content": "go"}, TOOL_USE, RESULTS,
               {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
    spans = _spans(history + [{"role": "user", "content": "next"}, REMINDER])
    assert _tools(spans) == {}, "an assistant turn with no tool_use emits no tool span"
    prompts = [s for s in spans if s.name.startswith("Claude_Code_Internal_Prompt_")]
    assert len(prompts) <= 3, "history before the last assistant message is not re-emitted"
