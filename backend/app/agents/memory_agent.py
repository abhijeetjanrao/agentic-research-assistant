"""
Memory Agent.

Role in the system:
    This is what fulfills the "manager remembers previous steps during a
    research session" requirement. Rather than relying on implicit
    context (every agent just reading the full ResearchState directly),
    the Memory Agent explicitly distills state["agent_trace"] into a
    short, human-readable summary and writes it to
    state["memory_context"]. Planner and Retriever prompts include this
    string, so later-in-session decisions are informed by earlier ones
    (e.g. the planner won't re-plan a step already completed; the
    retriever's query can be phrased with awareness of what's already
    been found).

Why a dedicated agent instead of just passing agent_trace directly into
prompts:
    agent_trace is a structured list of dicts -- fine for programmatic
    use, but wasteful and noisy as raw context to hand an LLM. This
    agent's job is specifically the compression step: turn "5 structured
    trace entries" into "2-3 sentences an LLM can use as context." It's
    also independently testable and swappable (e.g. later we could make
    this agent itself an LLM call for smarter summarization; today it's
    deterministic string formatting, which is simpler and sufficient
    for a bounded, short-lived session trace).
"""

from app.graph.state import AgentStep, ResearchState


def _describe_step(step: AgentStep) -> str:
    return f"- {step['agent_name']}: {step['summary']}"


def memory_node(state: ResearchState) -> dict:
    trace = state["agent_trace"]

    if not trace:
        memory_context = "No prior steps taken yet in this session."
    else:
        lines = [_describe_step(step) for step in trace]
        memory_context = "Steps taken so far in this session:\n" + "\n".join(lines)

    trace_entry: AgentStep = {
        "agent_name": "memory_agent",
        "summary": f"Condensed {len(trace)} prior step(s) into memory context",
        "data": {},
    }

    return {"memory_context": memory_context, "agent_trace": [trace_entry]}
