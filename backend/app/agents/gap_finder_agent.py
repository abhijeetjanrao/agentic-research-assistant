"""
Gap Finder Agent.

This is the agent that makes the Module 4 "star topology" loop-back
design actually mean something: instead of a fixed pipeline that always
ends at the report, this agent can send the Manager back to the
Retriever when it judges the research genuinely incomplete.

Loop contract (confirmed requirements):
    - If important gaps are found, re-trigger retrieval (local FAISS +
      web search if still needed) using a targeted follow-up query.
    - Cap retries at MAX_GAP_RETRIES (2), tracked via
      state["gap_retry_count"], to guarantee termination.
    - If gaps remain after the retry cap is reached, do NOT loop again --
      proceed to the report with the remaining gaps recorded in
      state["gaps"] so the Report Generator (Module 7) can surface them
      as a documented limitation, rather than silently dropping them or
      looping forever.

Why the LLM is asked for a follow-up search query, not just a gap
description:
    A gap description like "unclear how RAG compares to fine-tuning cost"
    isn't itself a good retrieval query. Asking the model for both --
    the gap AND a targeted query to fill it -- means the next Retriever
    pass searches for something more specific than the original query,
    which is the actual point of looping (a smarter second attempt, not
    a repeat of the first).
"""

import json
import re

from app.agents.llm import get_llm
from app.graph.state import AgentStep, ResearchState
from app.logging_config import logger

MAX_GAP_RETRIES = 2

GAP_FINDER_SYSTEM_PROMPT = """You are the Gap Finder Agent in a multi-agent research assistant.
You will be given the original research plan and the summary produced so far.
Identify any IMPORTANT gaps: significant plan items that are unanswered or only
weakly covered by the summary. Do not flag minor or stylistic issues -- only
gaps that would meaningfully mislead or under-inform someone reading the report.

Respond with ONLY a JSON object, nothing else. No markdown, no explanation.
Format:
{"gaps": ["<short description of gap 1>", "..."], "follow_up_query": "<a specific search query that would help fill the most important gap, or null if gaps is empty>"}

If there are no important gaps, respond with: {"gaps": [], "follow_up_query": null}
"""


def _extract_json_object(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def gap_finder_node(state: ResearchState) -> dict:
    retry_count = state["gap_retry_count"]

    llm = get_llm()
    user_prompt = (
        f"Original plan:\n{state['plan']}\n\n"
        f"Summary produced so far:\n{state['summary']}\n\n"
        f"Identify important gaps now."
    )
    response = llm.invoke(
        [
            ("system", GAP_FINDER_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    try:
        parsed = _extract_json_object(response.content)
        gaps = parsed.get("gaps") or []
        follow_up_query = parsed.get("follow_up_query")
        if not isinstance(gaps, list):
            raise ValueError("'gaps' was not a list")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "Gap Finder LLM output was not valid JSON ({error}); assuming no gaps. Raw: {raw}",
            error=str(e),
            raw=response.content[:200],
        )
        gaps, follow_up_query = [], None

    should_retry = bool(gaps) and bool(follow_up_query) and retry_count < MAX_GAP_RETRIES

    if should_retry:
        logger.info(
            "Gap Finder found {n} gap(s); triggering retry {attempt}/{max} with query: {q}",
            n=len(gaps),
            attempt=retry_count + 1,
            max=MAX_GAP_RETRIES,
            q=follow_up_query,
        )
        trace_entry: AgentStep = {
            "agent_name": "gap_finder_agent",
            "summary": f"Found {len(gaps)} gap(s); retrying retrieval (attempt {retry_count + 1}/{MAX_GAP_RETRIES})",
            "data": {"gaps": gaps, "follow_up_query": follow_up_query},
        }
        return {
            # Reset synthesis outputs so they regenerate against the
            # expanded evidence once the follow-up retrieval completes.
            "summary": None,
            "comparison": None,
            "compare_considered": False,
            "citations": None,
            "gaps": None,  # None (not []) signals "gap-finder hasn't re-checked this pass yet"
            "gap_query": follow_up_query,
            "gap_retrieval_pending": True,
            "gap_retry_count": retry_count + 1,
            "web_search_attempted": False,  # allow web search again for the new, more targeted query
            "agent_trace": [trace_entry],
        }

    # Either no important gaps, or we've hit the retry cap -- stop looping.
    # `gaps` is set to whatever remains (possibly non-empty) so the Report
    # Generator can surface it as a documented limitation.
    reason = (
        "no important gaps found"
        if not gaps
        else f"retry cap ({MAX_GAP_RETRIES}) reached -- proceeding with remaining gaps noted"
    )
    trace_entry: AgentStep = {
        "agent_name": "gap_finder_agent",
        "summary": f"Gap check complete: {reason}",
        "data": {"gaps": gaps},
    }
    return {"gaps": gaps, "agent_trace": [trace_entry]}
