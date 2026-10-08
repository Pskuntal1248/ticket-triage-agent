"""Entry point for Gravity."""

import json
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
    # AGENT_STANDARD section 8 requires durable hosted tools. The SDK is provided
    # by Gravity, not a dependency. Off-platform only, use the documented MCP client.
    try:
        import gravity
    except ModuleNotFoundError as exc:
        if exc.name != "gravity":
            raise
        available = await client.get_tools()
    else:
        available = await gravity.tools()
    tools = {t.name: t for t in available}
    structured = model.with_structured_output(Classification)

    async def classifier(subject: str, body: str) -> Classification:
        result = await structured.ainvoke([
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": json.dumps({"subject": subject, "body": body})},
        ])
        assert isinstance(result, Classification)
        return result

    return build_graph(classifier, tools)
