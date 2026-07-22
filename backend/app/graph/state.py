"""
Shared LangGraph state schema.

Why this file is the most important design decision in the project:
    LangGraph passes one state object between every node in the graph.
    Instead of agents calling each other directly (tight coupling, hard
    to test in isolation, hard to add new agents later), each agent is a
    pure function: (state) -> partial_state_update. LangGraph merges
    updates into the running state and passes the result to the next
    node. This is what gives us:
      - Memory during a session: earlier agents' outputs (the plan,
        retrieved chunks, search results) are still in `state` when later
        agents run, without any agent needing a reference to another.
      - Testability: any agent node can be unit tested by constructing a
        fake ResearchState and calling the node function directly.
      - Extensibility: adding a new agent (Module 6, 7) means adding a
        new key to the state and a new node function -- no existing node
        needs to change.

Why TypedDict instead of a Pydantic model:
    LangGraph's StateGraph is built around dict-like state with reducer
    functions per key (see the `Annotated[..., operator.add]` pattern
    below for list fields that should accumulate rather than overwrite).
    TypedDict is the documented, idiomatic choice for LangGraph state.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentStep(TypedDict):
    """One record of 'an agent ran and produced this output' -- appended
    to state.agent_trace by every node. This is what lets us show the
    user (or a reviewer) the full reasoning trace, and it's what the
    Memory Agent (Module 5) will read to answer 'what have we found so
    far in this session.'"""

    agent_name: str
    summary: str
    data: Dict[str, Any]


class RetrievedChunk(TypedDict):
    document_id: int
    page_number: int
    text: str
    score: float
    source: str  # "local_document" or "web_search"


class ResearchState(TypedDict):
    # --- Session identity ---
    session_id: int
    query: str

    # --- Planner output ---
    plan: Optional[List[str]]  # ordered list of research sub-steps

    # --- Retrieval outputs (accumulated across retriever + web search) ---
    # Annotated with operator.add so multiple nodes appending to this list
    # in the same graph step get merged rather than one overwriting the
    # other -- important once Retriever and WebSearch agents can run in
    # parallel (Module 5).
    retrieved_chunks: Annotated[List[RetrievedChunk], operator.add]
    web_search_attempted: bool  # guards against re-triggering web search every manager loop

    # --- Memory (Module 5) ---
    # Condensed "what have we established so far" string, produced by the
    # Memory Agent from agent_trace, and injected into Planner/Retriever
    # prompts on later manager loops -- this is the actual mechanism that
    # gives the manager cross-step memory within a session.
    memory_context: Optional[str]

    # --- Synthesis outputs (Module 6) ---
    summary: Optional[str]
    comparison: Optional[str]
    compare_considered: bool  # True once the Compare Agent has run/decided it doesn't apply
    citations: Optional[List[Dict[str, Any]]]
    gaps: Optional[List[str]]
    gap_retry_count: int  # how many times we've looped back to Retriever due to a detected gap
    gap_query: Optional[str]  # if set, Retriever/WebSearch search this instead of the original query
    gap_retrieval_pending: bool  # True right after Gap Finder triggers a retry; cleared by Retriever once it runs

    # --- Final outputs (Module 7) ---
    literature_review: Optional[str]
    final_report: Optional[str]

    # --- Control flow ---
    next_agent: Optional[str]  # set by the manager to route the next node
    iteration_count: int  # guards against infinite retrieve<->gap-find loops

    # --- Full audit trail of every agent that ran ---
    agent_trace: Annotated[List[AgentStep], operator.add]


def create_initial_state(session_id: int, query: str) -> ResearchState:
    """Factory for a fresh state at the start of a research session's
    graph run. Centralizing this avoids every caller having to remember
    every field name and its correct empty default."""
    return ResearchState(
        session_id=session_id,
        query=query,
        plan=None,
        retrieved_chunks=[],
        web_search_attempted=False,
        memory_context=None,
        summary=None,
        comparison=None,
        compare_considered=False,
        citations=None,
        gaps=None,
        gap_retry_count=0,
        gap_query=None,
        gap_retrieval_pending=False,
        literature_review=None,
        final_report=None,
        next_agent=None,
        iteration_count=0,
        agent_trace=[],
    )
