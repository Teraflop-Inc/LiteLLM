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


def _index(span):
    a = span.attributes
    return a["llm.message.index"] if "llm.message.index" in a else a["llm.message.input_index"]


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
    # The window is exactly the last turn: the assistant message (1) and the two user messages
    # after it, the reminder whole.
    prompts = [s for s in spans if s.name.startswith("Claude_Code_Internal_Prompt_")]
    assert sorted(_index(s) for s in prompts) == [1, 2, 3]
    reminder = next(s for s in prompts if _index(s) == 3)
    assert json.loads(reminder.attributes["input.value"]) == REMINDER["content"]


def test_two_trailing_user_messages_still_emit_only_the_latest_turn():
    history = [{"role": "user", "content": "go"}, TOOL_USE, RESULTS,
               {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
    spans = _spans(history + [{"role": "user", "content": "next"}, REMINDER])
    assert _tools(spans) == {}, "an assistant turn with no tool_use emits no tool span"
    prompts = [s for s in spans if s.name.startswith("Claude_Code_Internal_Prompt_")]
    assert sorted(_index(s) for s in prompts) == [3, 4, 5], "no history before the last assistant"


def test_a_request_ending_in_an_assistant_message_keeps_the_last_two():
    spans = _spans([{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "go"}, {"role": "assistant", "content": "prefill"}])
    prompts = [s for s in spans if s.name.startswith("Claude_Code_Internal_Prompt_")]
    assert [(s.attributes["llm.message.input_index"], s.attributes["llm.message.output_index"])
            for s in prompts] == [(2, 3)], "the user prompt and the prefill, paired"


def test_the_window_is_capped_when_many_user_messages_follow():
    spans = _spans([{"role": "user", "content": "go"}, TOOL_USE, RESULTS] + [REMINDER] * 20)
    prompts = [s for s in spans if s.name.startswith("Claude_Code_Internal_Prompt_")]
    assert len(prompts) == 2, "past the cap, the window falls back to the last two messages"
