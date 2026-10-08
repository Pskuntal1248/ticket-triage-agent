"""Pure Python refund policy rules. No LLM, no I/O, fully testable."""

from __future__ import annotations

from relayg.state import Classification, PolicyDecision

AUTO_APPROVE_LIMIT = 50.0
HUMAN_APPROVAL_LIMIT = 200.0


def decide(classification: Classification) -> tuple[PolicyDecision, str]:
    """Apply refund policy to a classification.

    Rules:
      - not a refund request, or no amount extracted: no_refund
      - amount < $50: auto_approve
      - $50 <= amount <= $200: needs_approval (human in the loop)
      - amount > $200: escalate
    """
    amount = classification.refund_amount
    if classification.intent != "refund_request":
        return "no_refund", "Not an explicit refund request; draft a reply."
    if amount is None or amount <= 0:
        return "no_refund", "Refund amount is missing or invalid; manual review needed, no refund recommendation."
    if amount < AUTO_APPROVE_LIMIT:
        return "auto_approve", f"${amount:.2f} is under the ${AUTO_APPROVE_LIMIT:.0f} auto-approval limit."
    if amount <= HUMAN_APPROVAL_LIMIT:
        return (
            "needs_approval",
            f"${amount:.2f} is between ${AUTO_APPROVE_LIMIT:.0f} and ${HUMAN_APPROVAL_LIMIT:.0f}; a human must approve.",
        )
    return "escalate", f"${amount:.2f} exceeds ${HUMAN_APPROVAL_LIMIT:.0f}; escalating to a senior agent."
