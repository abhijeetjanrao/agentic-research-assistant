"""
Planner Agent.

Prompt engineering decisions:
    - We force strict JSON output (a plain list of strings) rather than
      free-form text. Downstream code (the manager, and eventually the
      Retriever which may execute plan steps one at a time) needs to
      iterate over discrete steps programmatically -- parsing prose back
      into steps would be brittle and unnecessary when we can just ask
      the model for the exact structure we need.
    - memory_context (from the Memory Agent) is injected into the prompt
      so a re-planning pass (if it ever happens) is aware of what's
      already been established, rather than starting blind every time.
    - We wrap the Gemini call with a defensive JSON parse: if the model
      wraps its output in markdown code fences (a common LLM habit even
      when told not to), we strip those before parsing rather than
      failing outright.
"""

import json
import re

from app.agents.llm import get_llm
from app.graph.state import AgentStep, ResearchState
from app.logging_config import logger

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent in a multi-agent research assistant.
Given a research query, break it down into 3-5 concrete, ordered research sub-steps.

Respond with ONLY a JSON array of strings, nothing else. No markdown, no explanation.
Example response format:
["Define key terms in the query", "Find primary sources on X", "Compare X and Y", "Identify open questions"]
"""


def _extract_json_array(raw_text: str) -> list:
    """Strip markdown code fences if present, then parse JSON. Models
    frequently wrap JSON in ```json ... ``` even when explicitly told
    not to -- handling that here is more robust than a stricter prompt
    alone."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def planner_node(state: ResearchState) -> dict:
    memory_context = state.get("memory_context") or "No prior context."

    user_prompt = (
        f"Research query: {state['query']}\n\n"
        f"Prior session context:\n{memory_context}\n\n"
        f"Produce the ordered research plan now."
    )

    llm = get_llm()
    response = llm.invoke(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    try:
        plan = _extract_json_array(response.content)
        if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
            raise ValueError("parsed JSON was not a list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "Planner LLM output was not valid JSON ({error}); falling back to single-step plan. Raw output: {raw}",
            error=str(e),
            raw=response.content[:200],
        )
        plan = [f"Research and answer: {state['query']}"]

    trace_entry: AgentStep = {
        "agent_name": "planner_agent",
        "summary": f"Generated a {len(plan)}-step research plan",
        "data": {"plan": plan},
    }

    return {"plan": plan, "agent_trace": [trace_entry]}
