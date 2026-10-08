"""Typed state shared by every node in the graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Intent = Literal["refund_request", "question", "complaint", "bug_report", "other"]
Urgency = Literal["low", "medium", "high"]
PolicyDecision = Literal["auto_approve", "needs_approval", "escalate", "no_refund"]


class Classification(BaseModel):
    """Structured output produced by the classifier (LLM or keyword fallback)."""

    intent: Intent = Field(description="The customer's primary intent.")
    urgency: Urgency = Field(description="How urgent the ticket is.")
    refund_amount: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Refund amount in USD if the customer explicitly asks for one, else null.",
    )


class TicketState(TypedDict, total=False):
    """State carried through the graph for a single ticket."""

    inputs: dict  # the consumer's form answers (required by Gravity)
    result: Any  # what the consumer gets (required by Gravity)
    ticket_id: str
    subject: str
    body: str
    customer_email: str
    escalation_channel: str
    classification: Classification
    classifier: str  # "model" or "keyword-fallback"
    policy_decision: PolicyDecision
    policy_reason: str
    input_valid: bool
    input_error: str
    approved: bool
    actions: list[dict]
    notes: list[str]
