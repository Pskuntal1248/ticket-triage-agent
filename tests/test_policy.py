from relayg.policy import decide
from relayg.state import Classification


def _c(intent: str = "refund_request", amount: float | None = None) -> Classification:
    return Classification(intent=intent, urgency="medium", refund_amount=amount)


def test_non_refund_intent_is_no_refund() -> None:
    assert decide(_c(intent="question", amount=100.0))[0] == "no_refund"


def test_refund_without_amount_is_no_refund() -> None:
    assert decide(_c(amount=None))[0] == "no_refund"


def test_under_50_auto_approves() -> None:
    assert decide(_c(amount=49.99))[0] == "auto_approve"
    assert decide(_c(amount=0.01))[0] == "auto_approve"


def test_50_to_200_needs_approval() -> None:
    assert decide(_c(amount=50.0))[0] == "needs_approval"
    assert decide(_c(amount=120.0))[0] == "needs_approval"
    assert decide(_c(amount=200.0))[0] == "needs_approval"


def test_over_200_escalates() -> None:
    assert decide(_c(amount=200.01))[0] == "escalate"
    assert decide(_c(amount=450.0))[0] == "escalate"
