"""
Literature Review Agent.

Why this is a separate step from the Report Generator rather than folded
into it:
    A literature review is a specific genre with its own expected
    structure (background/context -> synthesized findings -> comparative
    analysis if relevant -> open questions), distinct from "assemble a
    document with a title and a citations section." Splitting them means:
      - This agent's only job is narrative structure and academic tone.
      - report_agent.py's only job is document assembly (numbering
        citations, adding a limitations section from gaps, persisting to
        the DB) -- it doesn't also need to reason about prose structure.
    This separation of concerns is what "Literature Review Agent" and
    "Report Generator Agent" being two distinct boxes in the original
    architecture diagram actually means in code.

Prompt engineering decision: the comparison section is only included in
the prompt if state["comparison"] is not None (Compare Agent decided the
query wasn't comparative) -- so the model isn't asked to awkwardly
reference a comparison that doesn't exist.
"""

from app.agents.llm import get_llm
from app.graph.state import AgentStep, ResearchState

LITERATURE_REVIEW_SYSTEM_PROMPT = """You are the Literature Review Agent in a multi-agent research assistant.
Given a research query, a synthesized summary, and (optionally) a comparison,
write a literature-review-style narrative with this structure:

## Background
Brief framing of the research question.

## Key Findings
The synthesized findings from the summary, organized coherently.

## Comparative Analysis
(Only include this section if a comparison was provided.)

## Open Questions
Note any inherent ambiguities or areas warranting further investigation
based on the findings themselves (not gap-finder output -- that's handled
separately in the final report).

Write in clear academic prose. Do not fabricate sources or findings beyond
what's in the provided summary/comparison.
"""


def literature_review_node(state: ResearchState) -> dict:
    llm = get_llm()

    comparison_section = (
        f"\n\nComparison:\n{state['comparison']}" if state.get("comparison") else ""
    )
    user_prompt = (
        f"Research query: {state['query']}\n\n"
        f"Summary:\n{state['summary']}"
        f"{comparison_section}\n\n"
        f"Write the literature review now."
    )

    response = llm.invoke(
        [
            ("system", LITERATURE_REVIEW_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )
    literature_review = response.content.strip()

    trace_entry: AgentStep = {
        "agent_name": "literature_review_agent",
        "summary": "Composed literature-review-style narrative",
        "data": {},
    }

    return {"literature_review": literature_review, "agent_trace": [trace_entry]}
