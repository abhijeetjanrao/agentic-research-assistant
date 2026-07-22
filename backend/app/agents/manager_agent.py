"""
Manager Agent.

Role in the system:
    The Manager is not a worker agent that produces research content --
    it's the orchestrator. Every time control returns to it, it looks at
    the current ResearchState and decides which worker agent should run
    next. This is implemented as a LangGraph "router" node: it returns a
    state update (just `next_agent` and an audit trace entry) rather than
    doing any research work itself.

Routing logic (Module 6 update):
    1. memory        -- once, at the start of a session.
    2. planner       -- once memory_context exists but no plan yet.
    3. retriever     -- once a plan exists but no chunks retrieved yet,
                        OR the Gap Finder just triggered a follow-up
                        retrieval (gap_retrieval_pending).
    4. websearch     -- ONLY if local retrieval is thin (per
                        retriever_agent.retrieval_is_sufficient) AND we
                        haven't already tried web search this pass.
    5. summarizer    -- once retrieval for this pass is done and no
                        summary yet.
    6. compare       -- once summarized but not yet considered for
                        comparison (compare_considered flag).
    7. citation      -- once compared/considered but no citations yet.
    8. gap_finder    -- once citations exist but gaps haven't been
                        (re-)checked yet this pass (gaps is None).
                        May loop back to step 3 by setting
                        gap_retrieval_pending + resetting synthesis
                        outputs -- capped at MAX_GAP_RETRIES by
                        gap_finder_agent.py itself.
    9. report_generator -- once gaps are known (checked, whether or not
                        any remain) and no final report yet.
    10. end          -- once final report exists.

Why routing logic lives here instead of being hardcoded into the graph
edges directly:
    Keeping the decision in a plain Python function (rather than encoding
    it purely as static graph edges) is what lets the manager make
    context-dependent decisions -- like the gap-driven loop-back -- 
    without changing the graph's structure (research_graph.py) at all.
"""

from app.agents.retriever_agent import retrieval_is_sufficient
from app.graph.state import AgentStep, ResearchState
from app.logging_config import logger

# Raised from Module 5's 8 to comfortably fit the worst case: memory + planner
# + up to 3 retrieval passes (initial + 2 gap retries) each potentially with
# websearch + summarizer + compare + citation + gap_finder, plus the final report.
MAX_ITERATIONS = 20


def manager_node(state: ResearchState) -> dict:
    """Decide the next agent to run based on what's already in state.

    Returns a partial state update: `next_agent` (consumed by the
    conditional edge function below) and an `agent_trace` entry recording
    the decision, for auditability.
    """
    iteration = state["iteration_count"] + 1

    if iteration > MAX_ITERATIONS:
        decision = "report_generator"
        reason = f"iteration ceiling ({MAX_ITERATIONS}) reached -- forcing report generation"
    elif state["memory_context"] is None:
        decision = "memory"
        reason = "no memory context established yet this session"
    elif state["plan"] is None:
        decision = "planner"
        reason = "no plan yet"
    elif not state["retrieved_chunks"] or state["gap_retrieval_pending"]:
        decision = "retriever"
        reason = (
            "gap finder triggered a follow-up retrieval"
            if state["gap_retrieval_pending"]
            else "plan exists but no chunks retrieved yet"
        )
    elif not retrieval_is_sufficient(state) and not state["web_search_attempted"]:
        decision = "websearch"
        reason = "local retrieval was thin/low-confidence -- falling back to web search"
    elif state["summary"] is None:
        decision = "summarizer"
        reason = "retrieval complete but not yet summarized"
    elif not state["compare_considered"]:
        decision = "compare"
        reason = "summary ready -- checking whether a comparison applies"
    elif state["citations"] is None:
        decision = "citation"
        reason = "generating citations from retrieved evidence"
    elif state["gaps"] is None:
        decision = "gap_finder"
        reason = "checking for important research gaps"
    elif state["literature_review"] is None:
        decision = "lit_review"
        reason = "gaps resolved (or retry cap reached) -- composing literature review"
    elif state["final_report"] is None:
        decision = "report_generator"
        reason = "literature review ready, assembling and persisting final report"
    else:
        decision = "end"
        reason = "final report already produced"

    logger.info(
        "Manager routing decision: {decision} ({reason})",
        decision=decision,
        reason=reason,
    )

    trace_entry: AgentStep = {
        "agent_name": "manager_agent",
        "summary": f"Routed to '{decision}': {reason}",
        "data": {"decision": decision, "iteration": iteration},
    }

    return {
        "next_agent": decision,
        "iteration_count": iteration,
        "agent_trace": [trace_entry],
    }


def route_after_manager(state: ResearchState) -> str:
    """Conditional edge function: LangGraph calls this after manager_node
    runs, and its return value selects which node to visit next. Kept as
    a thin wrapper around state["next_agent"] (rather than re-deriving
    the decision) so there's exactly one place the routing logic lives."""
    return state["next_agent"]
