# ticket-triage

A support ticket triage agent for the Gravity platform (LangGraph).
It classifies a ticket with the model, applies fixed refund rules in plain Python, and drafts or sends the reply.

| Refund amount | What happens |
| --- | --- |
| not a refund request | reply drafted |
| under $50 | auto-approved by policy, reply drafted, refund recorded as a recommendation |
| $50 to $200 | reply is sent only after a person approves it (gated `gmail.send`); not approved leaves a draft |
| over $200 | escalated to Slack (gated `slack.send`, optional channel) and a reply drafted |

Gravity has no refund tool, so the agent records the refund as a recommendation; it does not move money.
The refund amount must appear in the ticket text, so a number the model makes up is dropped or corrected.

## Credit

Adapted from [RelayG](https://github.com/AgentPostmortem/RelayG) by Royal Simpson Pinto, MIT License (see `src/LICENSE`).
RelayG is described by its author as a code-first rebuild of "Resolvd". See `src/NOTICE` for what was changed.
The policy rules (`src/relayg/policy.py`) and the policy tests are RelayG's, unchanged apart from a notice.

## Layout

- `manifest.yaml` Gravity manifest (tools: gmail.draft, gmail.send, slack.send; approvals on both sends)
- `src/agent.py` entrypoint (`graph`), `src/relayg/` the adapted code
- `tests/` unit tests (fake tools, no gateway needed): `pytest tests`

## Build

```bash
uv tool install gravity-cli
gravity validate && gravity build   # push needs `gravity login`
```
