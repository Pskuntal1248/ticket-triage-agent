"""Wire the state machine: prepare -> classify -> policy_check -> act -> finish."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from relayg.nodes import Classifier, finish, make_act, make_classify, make_prepare, policy_check
from relayg.state import TicketState
from relayg.tools import Tool


def build_graph(classifier: Classifier, tools: dict[str, Tool]):
    builder = StateGraph(TicketState)
    builder.add_node("prepare", make_prepare())
    builder.add_node("classify", make_classify(classifier))
    builder.add_node("policy_check", policy_check)
    builder.add_node("act", make_act(tools))
    builder.add_node("finish", finish)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "classify")
    builder.add_edge("classify", "policy_check")
    builder.add_edge("policy_check", "act")
    builder.add_edge("act", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
