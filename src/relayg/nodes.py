"""Graph nodes: classify, policy_check, act.

Adapted from RelayG (https://github.com/AgentPostmortem/RelayG), Copyright (c) 2026 Royal Simpson Pinto, MIT License.
Modified: the classifier calls the Gravity LLM gateway (no Groq); the human approval step is an
approval-gated gmail.send / slack.send call instead of langgraph interrupt(); actions use Gravity
tools instead of mock tools writing audit.jsonl; the refund amount is checked against the ticket text.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from relayg.policy import decide
from relayg.state import Classification, TicketState
from relayg.tools import Tool, ToolRefused, payload, reply_message, was_declined

Classifier = Callable[[str, str], Awaitable[Classification]]

_AMOUNT_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)")

CLASSIFIER_PROMPT = (
    "You are a support ticket triage classifier. Classify this ticket.\n"
    "Extract the refund amount in USD only if the customer explicitly asks "
    "for a refund; otherwise leave it null. Use only what is in the ticket.\n\n"
    "Subject: {subject}\n\nBody: {body}"
)


def mock_classify(subject: str, body: str) -> Classification:
    """Deterministic keyword classifier (RelayG's mock). Used as a named fallback and in tests."""
    text = f"{subject} {body}".lower()
    amount_match = _AMOUNT_RE.search(f"{subject} {body}")
    amount = float(amount_match.group(1)) if amount_match else None

    if "refund" in text or "money back" in text or "charge" in text:
        intent: str = "refund_request"
    elif "bug" in text or "error" in text or "crash" in text:
        intent = "bug_report"
    elif "angry" in text or "unacceptable" in text or "terrible" in text:
        intent = "complaint"
    elif "?" in body or "how" in text or "where" in text:
        intent = "question"
    else:
        intent = "other"

    if "urgent" in text or "asap" in text or "immediately" in text:
        urgency: str = "high"
    elif intent in ("complaint", "refund_request"):
        urgency = "medium"
    else:
        urgency = "low"

    refund_amount = amount if intent == "refund_request" else None
    return Classification(intent=intent, urgency=urgency, refund_amount=refund_amount)


def amounts_in(subject: str, body: str) -> list[float]:
    return [float(m) for m in _AMOUNT_RE.findall(f"{subject} {body}")]


def ground_amount(c: Classification, subject: str, body: str) -> tuple[Classification, str | None]:
    """Never trust a model-invented number: the refund amount must appear in the ticket text."""
    if c.refund_amount is None:
        return c, None
    found = amounts_in(subject, body)
    if any(abs(c.refund_amount - a) < 0.005 for a in found):
        return c, None
    if c.intent == "refund_request" and len(found) == 1:
        return c.model_copy(update={"refund_amount": found[0]}), (
            f"Model amount ${c.refund_amount:.2f} not in ticket; used ${found[0]:.2f} from the text."
        )
    return c.model_copy(update={"refund_amount": None}), (
        f"Model amount ${c.refund_amount:.2f} not found in ticket text; dropped it."
    )


def make_prepare() -> Callable[[TicketState], dict]:
    def prepare(state: TicketState) -> dict:
        i = state.get("inputs", {}) or {}
        return {
            "ticket_id": str(i.get("ticket_id", "")).strip() or "unknown",
            "subject": str(i.get("subject", "")),
            "body": str(i.get("body", "")),
            "customer_email": str(i.get("customer_email", "")).strip(),
            "escalation_channel": str(i.get("escalation_channel", "") or "").strip(),
        }

    return prepare


def make_classify(classifier: Classifier) -> Callable[[TicketState], Awaitable[dict]]:
    async def classify(state: TicketState) -> dict:
        subject, body = state.get("subject", ""), state.get("body", "")
        notes: list[str] = []
        try:
            c = await classifier(subject, body)
            source = "model"
        except Exception as exc:  # fail loudly in the result, then fall back
            notes.append(f"Model classification failed ({str(exc)[:120]}); used keyword fallback.")
            c = mock_classify(subject, body)
            source = "keyword-fallback"
        c, note = ground_amount(c, subject, body)
        if note:
            notes.append(note)
        return {"classification": c, "classifier": source, "notes": notes}

    return classify


def policy_check(state: TicketState) -> dict:
    """Node: apply pure Python refund policy rules."""
    decision, reason = decide(state["classification"])
    return {"policy_decision": decision, "policy_reason": reason}


def _money(c: Classification) -> str:
    return f"${(c.refund_amount or 0.0):.2f}"


def make_act(tools: dict[str, Tool]) -> Callable[[TicketState], Awaitable[dict]]:
    """Node: take the action the policy allows.

    Gated calls (gmail.send, slack.send) come first in this node: when the run resumes after an
    approval the node restarts, and the gateway answers repeated calls with the stored result.
    """

    async def act(state: TicketState) -> dict:
        decision = state["policy_decision"]
        c = state["classification"]
        subject = state.get("subject", "")
        email = state.get("customer_email", "")
        tid = state["ticket_id"]
        actions: list[dict] = []
        notes: list[str] = []
        update: dict[str, Any] = {}

        async def draft(text: str, label: str) -> None:
            if not email:
                actions.append({"action": label, "status": "skipped", "reason": "no customer email given"})
                return
            try:
                payload(await tools["gmail_draft"].ainvoke(reply_message(email, subject, text)))
                actions.append({"action": label, "status": "drafted", "to": email})
            except ToolRefused as exc:
                notes.append(f"gmail.draft failed: {exc}")
                actions.append({"action": label, "status": "failed", "error": str(exc)})

        if decision == "no_refund":
            await draft(
                f"Hi,\n\nThanks for reaching out about ticket {tid}. A support agent will reply shortly.\n",
                "draft_reply",
            )
        elif decision == "auto_approve":
            actions.append({
                "action": "refund_recommendation", "status": "recorded", "amount": c.refund_amount,
                "approved_by": "policy:auto", "note": "Issue the refund in your billing system; Gravity has no refund tool.",
            })
            await draft(
                f"Hi,\n\nYour refund request of {_money(c)} for ticket {tid} has been approved and is being processed.\n",
                "draft_reply",
            )
        elif decision == "needs_approval":
            # Human in the loop: gmail.send is approval-gated, the gateway holds the call until a person decides.
            if not email:
                update["approved"] = False
                notes.append("Needs approval but no customer email was given; nothing sent.")
                actions.append({"action": "send_reply", "status": "skipped", "reason": "no customer email given"})
            else:
                msg = reply_message(
                    email, subject,
                    f"Hi,\n\nYour refund request of {_money(c)} for ticket {tid} has been approved by our team.\n",
                )
                raw = await tools["gmail_send"].ainvoke(msg)
                if was_declined(raw):
                    update["approved"] = False
                    actions.append({"action": "send_reply", "status": "not approved", "amount": c.refund_amount})
                    await draft(
                        f"Hi,\n\nYour refund request of {_money(c)} for ticket {tid} is under review by our team.\n",
                        "draft_holding_reply",
                    )
                else:
                    try:
                        payload(raw)
                        update["approved"] = True
                        actions.append({
                            "action": "refund_recommendation", "status": "recorded", "amount": c.refund_amount,
                            "approved_by": "human", "note": "Issue the refund in your billing system; Gravity has no refund tool.",
                        })
                        actions.append({"action": "send_reply", "status": "sent", "to": email})
                    except ToolRefused as exc:
                        update["approved"] = False
                        notes.append(f"gmail.send failed: {exc}")
                        actions.append({"action": "send_reply", "status": "failed", "error": str(exc)})
        else:  # escalate
            channel = state.get("escalation_channel", "")
            if channel:
                text = (
                    f"Escalation: ticket {tid} asks for a refund of {_money(c)} "
                    f"(urgency {c.urgency}). {state['policy_reason']}"
                )
                raw = await tools["slack_send"].ainvoke({"channel": channel, "markdown_text": text})
                if was_declined(raw):
                    actions.append({"action": "escalate", "status": "not approved", "channel": channel})
                else:
                    try:
                        payload(raw)
                        actions.append({"action": "escalate", "status": "sent", "channel": channel})
                    except ToolRefused as exc:
                        notes.append(f"slack.send failed: {exc}")
                        actions.append({"action": "escalate", "status": "failed", "error": str(exc)})
            else:
                actions.append({"action": "escalate", "status": "skipped", "reason": "no escalation channel given"})
            await draft(
                f"Hi,\n\nYour request for ticket {tid} has been escalated to a senior agent.\n",
                "draft_reply",
            )

        update["actions"] = actions
        update["notes"] = notes
        return update

    return act


def finish(state: TicketState) -> dict:
    c = state["classification"]
    actions = state.get("actions", [])
    return {
        "result": {
            "ticket_id": state["ticket_id"],
            "intent": c.intent,
            "urgency": c.urgency,
            "refund_amount": c.refund_amount,
            "policy_decision": state["policy_decision"],
            "policy_reason": state["policy_reason"],
            "approved": state.get("approved"),
            "classifier": state.get("classifier"),
            "actions": actions,
            "notes": state.get("notes", []),
        }
    }
