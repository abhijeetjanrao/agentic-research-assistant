"""
Tests for the LangGraph state schema.

Full end-to-end graph execution tests (with every LLM/FAISS/Tavily
boundary mocked) now live in tests/test_module6_agents.py, since a
complete graph run exercises the synthesis and gap-finder nodes added in
that module -- keeping one authoritative set of full-graph tests avoids
two files drifting out of sync with the graph's actual node set.
"""

from app.graph.state import create_initial_state


def test_create_initial_state_defaults():
    state = create_initial_state(session_id=1, query="What is RAG?")
    assert state["session_id"] == 1
    assert state["query"] == "What is RAG?"
    assert state["plan"] is None
    assert state["retrieved_chunks"] == []
    assert state["web_search_attempted"] is False
    assert state["memory_context"] is None
    assert state["summary"] is None
    assert state["comparison"] is None
    assert state["compare_considered"] is False
    assert state["citations"] is None
    assert state["gaps"] is None
    assert state["gap_retry_count"] == 0
    assert state["gap_query"] is None
    assert state["gap_retrieval_pending"] is False
    assert state["iteration_count"] == 0
    assert state["agent_trace"] == []
