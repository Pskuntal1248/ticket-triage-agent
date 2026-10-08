"""Typed state shared by every node in the graph.

Adapted from RelayG (https://github.com/AgentPostmortem/RelayG), Copyright (c) 2026 Royal Simpson Pinto, MIT License.
Modified: added `inputs`, `result`, `classifier` and `notes` for the Gravity runtime.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

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
    approved: bool
    actions: Annotated[list[dict], operator.add]
    notes: Annotated[list[str], operator.add]
