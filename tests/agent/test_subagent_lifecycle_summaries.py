"""Public lifecycle summaries keep the native budget and recoverable spill path."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleService,
    SubagentResult,
    SubagentState,
)
from hermes_cli.plugin_host_wire import decode, encode

FULL_SUMMARY = "Opening finding\n" + "important evidence in the middle\n" * 2000 + "Closing recommendation\n"


class SummaryChild:
    tool_progress_callback = None
    _delegate_saved_tool_names = []
    _credential_pool = None
    _delegate_role = "leaf"
    _delegate_depth = 1
    _parent_subagent_id = None
    provider = "test"
    model = "test-model"
    session_prompt_tokens = session_completion_tokens = session_reasoning_tokens = 0
    session_estimated_cost_usd = 0.0
    _subagent_id = "sa-summary-fidelity"

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, **_kwargs):
        return {"final_response": FULL_SUMMARY, "completed": True, "api_calls": 1, "messages": []}

    def close(self):
        pass


@pytest.fixture
def harness(monkeypatch):
    # No compressor/last-usage attributes: there is no dynamic headroom cap in these cases.
    owner = SimpleNamespace(parent=SimpleNamespace(
        session_id="summary-owner", enabled_toolsets=None, _delegate_depth=0,
        _current_task_id=None, _touch_activity=lambda _description: None,
    ))
    child = SummaryChild()
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child)
    return SimpleNamespace(owner=owner, service=SubagentLifecycleService(lambda: owner.parent))


def finished(harness, monkeypatch, cap, wire):
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"max_summary_chars": cap})
    request = SubagentLaunchRequest(goal="Summarize the evidence")
    handle = harness.service.launch(decode(encode(request)) if wire else request)
    assert harness.service.wait(handle, timeout_seconds=10).state is SubagentState.SUCCEEDED
    result = harness.service.result(handle)
    assert result == harness.service.result(handle)  # retained result is idempotent
    return handle, decode(encode(result)) if wire else result


@pytest.mark.parametrize("wire", [False, True])
def test_no_second_summary_cap_when_native_limits_are_disabled(harness, monkeypatch, wire):
    _, result = finished(harness, monkeypatch, 0, wire)
    assert result.summary == FULL_SUMMARY
    assert result.summary_truncated is False and result.summary_full_path is None
    assert result.result_hash


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("cap", [24_000, 40_000])
def test_native_spill_metadata_and_footer_survive(harness, monkeypatch, wire, cap):
    _, result = finished(harness, monkeypatch, cap, wire)
    assert result.summary_truncated is True
    path = Path(result.summary_full_path)
    assert path.is_absolute() and path.read_text() == FULL_SUMMARY
    assert result.summary.startswith("Opening finding") and "Closing recommendation" in result.summary
    assert "[SUMMARY TRUNCATED]" in result.summary and str(path) in result.summary
    assert "read_file" in result.summary
    if cap == 40_000:
        assert len(result.summary) > 32_000  # a second lifecycle clip would discard the footer


def test_spill_failure_is_explicit_without_a_false_recovery_path(harness, monkeypatch):
    monkeypatch.setattr("tools.delegate_tool_results._spill_summary_to_file", lambda *_args: None)
    _, result = finished(harness, monkeypatch, 24_000, False)
    assert result.summary_truncated is True and result.summary_full_path is None
    assert "Full output could not be stored to disk" in result.summary


def test_parent_headroom_budget_still_applies_when_static_cap_is_off(harness, monkeypatch):
    harness.owner.parent.context_compressor = SimpleNamespace(context_length=10_000, max_tokens=1000)
    harness.owner.parent._last_prompt_size_tokens = 9000
    _, result = finished(harness, monkeypatch, 0, False)
    assert result.summary_truncated is True and len(result.summary) < 10_000
    assert Path(result.summary_full_path).read_text() == FULL_SUMMARY


def test_foreign_parent_cannot_read_summary_or_spill_path(harness, monkeypatch):
    handle, result = finished(harness, monkeypatch, 24_000, False)
    assert result.summary_full_path
    harness.owner.parent = SimpleNamespace(session_id="unrelated-owner")
    denied = harness.service.result(handle)
    assert denied.terminal_state is SubagentState.UNKNOWN and denied.ready is False
    assert denied.summary is None and denied.summary_full_path is None


def test_optional_result_fields_preserve_legacy_positional_construction(harness, monkeypatch):
    handle, _ = finished(harness, monkeypatch, 0, False)
    result = SubagentResult(handle, SubagentState.SUCCEEDED, True, "legacy", None,
                            1.0, 2.0, None, None, {"api_calls": 1}, {}, "legacy-hash")
    assert result.result_hash == "legacy-hash"
    assert result.summary_truncated is False and result.summary_full_path is None
