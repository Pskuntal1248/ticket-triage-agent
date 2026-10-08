"""Tests for the ticket triage agent. Fake tools and a fake classifier: no gateway, model or disk needed."""
import asyncio
import json
import os

from relayg.graph import build_graph
from relayg.nodes import ground_amount, mock_classify
from relayg.state import Classification


class FakeTool:
    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply if reply is not None else json.dumps({"successful": True, "data": {"id": "1"}})

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.reply


def make_tools(send_reply=None, slack_reply=None):
    return {"gmail_draft": FakeTool(), "gmail_send": FakeTool(send_reply), "slack_send": FakeTool(slack_reply)}


async def kw_classifier(subject, body):
    return mock_classify(subject, body)


def run(subject, body, tools, classifier=kw_classifier, channel=""):
    g = build_graph(classifier, tools)
    inputs = {"ticket_id": "T-1", "subject": subject, "body": body, "customer_email": "c@example.com",
              "escalation_channel": channel}
    out = asyncio.run(g.ainvoke({"inputs": inputs}))
    return out["result"]


def test_small_refund_auto_approved_and_drafted():
    t = make_tools()
    r = run("Refund", "Please refund $20.", t)
    assert r["policy_decision"] == "auto_approve"
    assert [a["action"] for a in r["actions"]] == ["refund_recommendation", "draft_reply"]
    assert len(t["gmail_draft"].calls) == 1 and not t["gmail_send"].calls
    assert t["gmail_draft"].calls[0]["recipient_email"] == "c@example.com"


def test_question_gets_draft_only():
    t = make_tools()
    r = run("Help", "How do I reset my password?", t)
    assert r["policy_decision"] == "no_refund"
    assert [a["action"] for a in r["actions"]] == ["draft_reply"]


def test_mid_refund_sends_after_approval():
    t = make_tools()
    r = run("Refund", "Please refund $120.", t)
    assert r["policy_decision"] == "needs_approval"
    assert r["approved"] is True
    assert len(t["gmail_send"].calls) == 1 and not t["gmail_draft"].calls
    assert "$120.00" in t["gmail_send"].calls[0]["body"]


def test_mid_refund_not_approved_keeps_draft():
    t = make_tools(send_reply="APPROVAL_REJECTED: 'gmail_send' was not approved, so it did not run")
    r = run("Refund", "Refund $75 please.", t)
    assert r["approved"] is False
    assert [a["status"] for a in r["actions"]] == ["not approved", "drafted"]
    assert len(t["gmail_draft"].calls) == 1


def test_approval_timeout_treated_as_not_approved():
    t = make_tools(send_reply="APPROVAL_TIMEOUT")
    assert run("Refund", "Refund $75 please.", t)["approved"] is False


def test_send_failure_is_reported_not_hidden():
    t = make_tools(send_reply="gateway exploded")
    r = run("Refund", "Refund $75 please.", t)
    assert r["approved"] is False
    assert r["actions"][0]["status"] == "failed"
    assert any("gmail.send failed" in n for n in r["notes"])


def test_large_refund_escalates_to_slack():
    t = make_tools()
    r = run("Refund", "I want a refund of $450.", t, channel="C123")
    assert r["policy_decision"] == "escalate"
    assert [a["action"] for a in r["actions"]] == ["escalate", "draft_reply"]
    assert t["slack_send"].calls[0]["channel"] == "C123"


def test_escalation_without_channel_is_named_as_skipped():
    t = make_tools()
    r = run("Refund", "I want a refund of $450.", t)
    assert r["actions"][0]["status"] == "skipped"
    assert not t["slack_send"].calls


def test_model_invented_amount_is_corrected_from_ticket_text():
    async def lying(subject, body):
        return Classification(intent="refund_request", urgency="low", refund_amount=500.0)

    t = make_tools()
    r = run("Refund", "Please refund $20.", t, classifier=lying)
    assert r["refund_amount"] == 20.0
    assert r["policy_decision"] == "auto_approve"
    assert any("not in ticket" in n for n in r["notes"])


def test_model_amount_with_no_matching_text_is_dropped():
    c = Classification(intent="refund_request", urgency="low", refund_amount=99.0)
    fixed, note = ground_amount(c, "Refund", "I want my money back, $10 or $30.")
    assert fixed.refund_amount is None and note


def test_classifier_failure_falls_back_and_says_so():
    async def broken(subject, body):
        raise RuntimeError("llm down")

    t = make_tools()
    r = run("Refund", "Please refund $20.", t, classifier=broken)
    assert r["classifier"] == "keyword-fallback"
    assert any("failed" in n for n in r["notes"])
    assert r["policy_decision"] == "auto_approve"


def test_missing_customer_email_is_reported():
    t = make_tools()
    g = build_graph(kw_classifier, t)
    out = asyncio.run(g.ainvoke({"inputs": {"ticket_id": "T", "subject": "Help", "body": "How?"}}))
    assert out["result"]["actions"][0]["status"] == "skipped"
    assert not t["gmail_draft"].calls


def test_same_input_same_result_and_no_disk_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = run("Refund", "Please refund $120.", make_tools())
    b = run("Refund", "Please refund $120.", make_tools())
    assert a == b
    assert list(tmp_path.iterdir()) == []


def test_licence_files_ship_with_source():
    here = os.path.dirname(os.path.dirname(__file__))
    lic = open(os.path.join(here, "src", "LICENSE"), encoding="utf-8").read()
    assert "MIT License" in lic
