"""
Report Generator Agent.

Replaces the Module 4 stub. This is the final node in the graph's happy
path: it assembles the literature review, numbered citations, and a
limitations section (from any gaps the Gap Finder couldn't resolve
within its retry cap) into one markdown document -- then persists it to
MySQL immediately, via Module 2's crud.create_report().

Why persist here (in the agent) rather than leaving it to the API route
that calls the graph (confirmed requirement):
    A completed research session should be durably saved the instant the
    graph finishes, not conditional on whatever calling code happens to
    do next. If the API route crashed right after graph.invoke() returned
    but before it got around to saving, the finished report would be
    lost despite all the work being done. Saving inside the agent makes
    "graph finished" and "report is durably stored" the same event.

Why a DB failure here must NOT crash the whole graph run:
    The user should still get their finished report even if, say, MySQL
    is briefly unreachable. We log the failure and note it in the agent
    trace, but state["final_report"] is still populated and returned --
    a temporary storage hiccup shouldn't destroy finished research work
    sitting in memory.
"""

import json
from datetime import datetime, timezone

from app.db import crud
from app.db.session import SessionLocal
from app.graph.state import AgentStep, ResearchState
from app.logging_config import logger


def _format_citations_section(citations: list) -> str:
    if not citations:
        return "No citations were generated (no evidence was retrieved)."
    lines = []
    for c in citations:
        if c["source_type"] == "local_document":
            lines.append(
                f"[{c['index']}] Local document #{c['document_id']}, page {c['page_number']}: "
                f"\"{c['excerpt']}...\""
            )
        else:
            lines.append(f"[{c['index']}] Web source: {c['url']} -- \"{c['excerpt']}...\"")
    return "\n".join(lines)


def _format_limitations_section(gaps: list) -> str:
    if not gaps:
        return "No significant gaps were identified in this research."
    lines = [f"- {gap}" for gap in gaps]
    return (
        "The following gaps could not be fully resolved within the research "
        "session's retry limit and should be considered when interpreting "
        "this report:\n" + "\n".join(lines)
    )


def report_generator_node(state: ResearchState) -> dict:
    citations = state.get("citations") or []
    gaps = state.get("gaps") or []
    generated_at = datetime.now(timezone.utc).isoformat()

    final_report = (
        f"# Research Report: {state['query']}\n\n"
        f"*Generated {generated_at} | Session #{state['session_id']}*\n\n"
        f"{state['literature_review']}\n\n"
        f"## Citations\n\n{_format_citations_section(citations)}\n\n"
        f"## Limitations\n\n{_format_limitations_section(gaps)}\n"
    )

    # Persist immediately -- a finished report should never depend on the
    # caller remembering to save it (see module docstring above).
    db_saved = False
    try:
        db = SessionLocal()
        try:
            crud.create_report(
                db,
                session_id=state["session_id"],
                title=f"Research Report: {state['query']}",
                content_markdown=final_report,
                citations_json=json.dumps(citations),
            )
            db_saved = True
        finally:
            db.close()
    except Exception as e:
        # Don't let a storage hiccup destroy finished work sitting in
        # memory -- the report is still returned in state either way.
        logger.error("Failed to persist report to MySQL: {error}", error=str(e))

    trace_entry: AgentStep = {
        "agent_name": "report_generator_agent",
        "summary": f"Generated final report ({'saved to DB' if db_saved else 'DB SAVE FAILED'})",
        "data": {"db_saved": db_saved, "num_citations": len(citations), "num_gaps": len(gaps)},
    }

    return {"final_report": final_report, "agent_trace": [trace_entry]}
