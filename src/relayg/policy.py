"""Pure Python refund policy rules. No LLM, no I/O, fully testable.

Adapted from RelayG (https://github.com/AgentPostmortem/RelayG), Copyright (c) 2026 Royal Simpson Pinto, MIT License.
Unchanged apart from this notice and the import path.
"""

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
    if classification.intent != "refund_request" or amount is None:
        return "no_refund", "Not a refund request; reply normally."
    if amount < AUTO_APPROVE_LIMIT:
        return "auto_approve", f"${amount:.2f} is under the ${AUTO_APPROVE_LIMIT:.0f} auto-approval limit."
    if amount <= HUMAN_APPROVAL_LIMIT:
        return (
            "needs_approval",
            f"${amount:.2f} is between ${AUTO_APPROVE_LIMIT:.0f} and ${HUMAN_APPROVAL_LIMIT:.0f}; a human must approve.",
        )
    return "escalate", f"${amount:.2f} exceeds ${HUMAN_APPROVAL_LIMIT:.0f}; escalating to a senior agent."
