"""
Tests for human_in_the_loop pattern.
Tests approval routing, rejection routing, and state update mechanics.
No Ollama required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hitl_pattern import check_human_decision, format_pending_action, ApprovalState
from langchain_core.messages import HumanMessage


def _state(decision: str, feedback: str = "") -> ApprovalState:
    return {
        "messages": [HumanMessage(content="test")],
        "pending_action": "Send email to client",
        "human_decision": decision,
        "human_feedback": feedback,
        "action_result": "",
    }


def test_approve_routes_to_execute():
    assert check_human_decision(_state("approve")) == "execute"


def test_reject_routes_to_end():
    assert check_human_decision(_state("reject")) == "end"


def test_modify_routes_to_replan():
    assert check_human_decision(_state("modify", "Make it shorter")) == "replan"


def test_empty_decision_routes_to_end():
    assert check_human_decision(_state("")) == "end"


def test_format_pending_action_no_feedback():
    result = format_pending_action("Delete all records")
    assert "Delete all records" in result
    assert "PENDING ACTION" in result


def test_format_pending_action_with_feedback():
    result = format_pending_action("Send email", "Make it shorter")
    assert "Make it shorter" in result
    assert "Previous feedback" in result


def test_case_insensitive_decision():
    assert check_human_decision(_state("APPROVE")) == "execute"
    assert check_human_decision(_state("Reject")) == "end"
    assert check_human_decision(_state("MODIFY")) == "replan"
