"""Entry point for Gravity: ticket-triage, adapted from RelayG (MIT, (c) 2026 Royal Simpson Pinto)."""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from relayg.graph import build_graph
from relayg.nodes import CLASSIFIER_PROMPT
from relayg.state import Classification


async def graph():
    # Read the three environment variables here, inside the function, never at import time.
    token = os.environ["GRAVITY_RUN_TOKEN"]
    model = ChatOpenAI(
        model="openai/gpt-6-luna",
        base_url=os.environ["GRAVITY_LLM_URL"],
        api_key=token,
        temperature=0,
    )
    client = MultiServerMCPClient({"gravity": {
        "transport": "streamable_http",
        "url": os.environ["GRAVITY_GATEWAY_URL"],
        "headers": {"Authorization": f"Bearer {token}"},
    }})
    tools = {t.name: t for t in await client.get_tools()}  # only the tools in manifest.yaml
    structured = model.with_structured_output(Classification)

    async def classifier(subject: str, body: str) -> Classification:
        result = await structured.ainvoke(CLASSIFIER_PROMPT.format(subject=subject, body=body))
        assert isinstance(result, Classification)
        return result

    return build_graph(classifier, tools)
