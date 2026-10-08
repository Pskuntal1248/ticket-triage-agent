"""Gravity tool helpers."""

from __future__ import annotations

import json
from typing import Any, Protocol


class Tool(Protocol):
    async def ainvoke(self, input: dict) -> Any: ...


class ToolRefused(RuntimeError):
    """The gateway refused or failed a tool call."""


def text_of(raw: Any) -> str:
    if isinstance(raw, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, default=str)


def was_declined(raw: Any) -> bool:
    """True when an approval-gated call was not approved (rejected or timed out)."""
    text = text_of(raw)
    return "APPROVAL_REJECTED" in text or "APPROVAL_TIMEOUT" in text


def payload(raw: Any) -> Any:
    """A tool result as data. Gateway refusals arrive as text, not JSON: raise, never pass them on."""
    if isinstance(raw, list):
        raw = text_of(raw)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise ToolRefused(raw[:300]) from None
    if isinstance(raw, dict) and raw.get("successful") is False:
        raise ToolRefused(str(raw.get("error"))[:300])
    if not isinstance(raw, dict) or not raw or raw.get("isError") is True or raw.get("error"):
        raise ToolRefused("Gateway returned an empty, invalid or error result")
    return raw.get("data", raw)


def reply_message(customer_email: str, subject: str, body: str) -> dict:
    return {"recipient_email": customer_email, "subject": f"Re: {subject}".strip(), "body": body}
