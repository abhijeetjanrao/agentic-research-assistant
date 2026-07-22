"""
Compare Agent.

Design decision: cheap heuristic gate before the LLM call. Most research
queries aren't comparisons ("what is RAG?"), and running a "compare X vs
Y" prompt against a non-comparative query would either produce a useless
comparison or force the model to invent a second thing to compare
against. So we check for comparison-signaling language first; only if
that heuristic fires do we spend an LLM call on it. This is simpler and
more predictable than asking the LLM itself "is this a comparison?" as a
separate call, and costs nothing when it's obviously not applicable.

state["compare_considered"] is set True either way -- it's what tells
the Manager "the Compare Agent has already made its decision for this
session," so it isn't invoked again on a later manager loop (e.g. after
a gap-driven retry).
"""

import re

from app.agents.llm import get_llm
from app.graph.state import AgentStep, ResearchState

_COMPARISON_SIGNALS = re.compile(
    r"\bvs\.?\b|\bversus\b|\bcompare\b|\bcomparison\b|difference between|differences between|"
    r"which is better|pros and cons",
    re.IGNORECASE,
)

COMPARE_SYSTEM_PROMPT = """You are the Compare Agent in a multi-agent research assistant.
The user's query asks for a comparison. Using the provided summary and retrieved
chunks, produce a structured comparison as a markdown table (or clearly labeled
sections if a table doesn't fit) covering the key dimensions of difference.
Base the comparison only on the provided content -- do not invent facts.
"""


def _query_requests_comparison(query: str) -> bool:
    return bool(_COMPARISON_SIGNALS.search(query))


def compare_node(state: ResearchState) -> dict:
    if not _query_requests_comparison(state["query"]):
        trace_entry: AgentStep = {
            "agent_name": "compare_agent",
            "summary": "Query is not comparative -- skipped",
            "data": {"applicable": False},
        }
        return {"comparison": None, "compare_considered": True, "agent_trace": [trace_entry]}

    llm = get_llm()
    user_prompt = (
        f"Query: {state['query']}\n\n"
        f"Summary so far:\n{state['summary']}\n\n"
        f"Produce the comparison now."
    )
    response = llm.invoke(
        [
            ("system", COMPARE_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    trace_entry: AgentStep = {
        "agent_name": "compare_agent",
        "summary": "Query was comparative -- generated a structured comparison",
        "data": {"applicable": True},
    }

    return {
        "comparison": response.content.strip(),
        "compare_considered": True,
        "agent_trace": [trace_entry],
    }
