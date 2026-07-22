"""
Tests for Module 6: Summary, Compare, Citation, and Gap Finder agents,
plus full graph executions proving the bounded gap-retry loop actually
terminates and surfaces remaining gaps in the final state.

Gemini/Tavily calls are mocked (no live API key in this sandbox) -- see
the README note for this module on live verification.
"""

from unittest.mock import MagicMock, patch

from app.agents.citation_agent import citation_node
from app.agents.compare_agent import _query_requests_comparison, compare_node
from app.agents.gap_finder_agent import MAX_GAP_RETRIES, gap_finder_node
from app.graph.research_graph import build_research_graph
from app.graph.state import create_initial_state


# --- Summary Agent (mocked LLM) ---

@patch("app.agents.summary_agent.get_llm")
def test_summarizer_node_calls_llm_with_chunks(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="A concise summary of the chunks.")
    mock_get_llm.return_value = mock_llm

    from app.agents.summary_agent import summarizer_node

    state = create_initial_state(session_id=1, query="What is RAG?")
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "RAG combines retrieval and generation.", "score": 0.9, "source": "local_document"},
    ]
    update = summarizer_node(state)

    assert update["summary"] == "A concise summary of the chunks."
    assert "RAG combines retrieval" in mock_llm.invoke.call_args[0][0][1][1]


# --- Compare Agent ---

def test_query_requests_comparison_detects_vs():
    assert _query_requests_comparison("RAG vs fine-tuning") is True
    assert _query_requests_comparison("compare transformers and mamba") is True
    assert _query_requests_comparison("what is RAG?") is False


def test_compare_node_skips_non_comparative_query():
    state = create_initial_state(session_id=1, query="What is RAG?")
    state["summary"] = "some summary"
    update = compare_node(state)

    assert update["comparison"] is None
    assert update["compare_considered"] is True


@patch("app.agents.compare_agent.get_llm")
def test_compare_node_generates_comparison_for_comparative_query(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="| Aspect | RAG | Fine-tuning |\n|---|---|---|")
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="RAG vs fine-tuning")
    state["summary"] = "some summary"
    update = compare_node(state)

    assert update["comparison"] is not None
    assert update["compare_considered"] is True


# --- Citation Agent (deterministic, no LLM) ---

def test_citation_node_builds_citations_from_local_and_web_chunks():
    state = create_initial_state(session_id=1, query="test")
    state["retrieved_chunks"] = [
        {"document_id": 5, "page_number": 3, "text": "Local content here.", "score": 0.9, "source": "local_document"},
        {"document_id": -1, "page_number": 0, "text": "Web title: web content (source: https://example.com)", "score": 0.7, "source": "web_search"},
    ]
    update = citation_node(state)

    citations = update["citations"]
    assert len(citations) == 2
    assert citations[0]["source_type"] == "local_document"
    assert citations[0]["document_id"] == 5
    assert citations[0]["page_number"] == 3
    assert citations[1]["source_type"] == "web_search"
    assert citations[1]["url"] == "https://example.com"


def test_citation_node_handles_empty_chunks():
    state = create_initial_state(session_id=1, query="test")
    update = citation_node(state)
    assert update["citations"] == []


# --- Gap Finder Agent (mocked LLM) ---

@patch("app.agents.gap_finder_agent.get_llm")
def test_gap_finder_no_gaps_stops_looping(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"gaps": [], "follow_up_query": null}')
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="test")
    state["plan"] = ["step 1"]
    state["summary"] = "a thorough summary"
    update = gap_finder_node(state)

    assert update["gaps"] == []
    assert "gap_retrieval_pending" not in update  # no retry triggered


@patch("app.agents.gap_finder_agent.get_llm")
def test_gap_finder_triggers_retry_when_gaps_found_and_under_cap(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"gaps": ["missing cost comparison"], "follow_up_query": "RAG vs fine-tuning cost"}'
    )
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="test")
    state["plan"] = ["step 1"]
    state["summary"] = "a partial summary"
    state["gap_retry_count"] = 0
    update = gap_finder_node(state)

    assert update["gaps"] is None  # reset so gap_finder re-checks after the retry
    assert update["gap_query"] == "RAG vs fine-tuning cost"
    assert update["gap_retrieval_pending"] is True
    assert update["gap_retry_count"] == 1
    assert update["summary"] is None  # reset to regenerate with new evidence


@patch("app.agents.gap_finder_agent.get_llm")
def test_gap_finder_stops_at_retry_cap_and_surfaces_remaining_gaps(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"gaps": ["still missing X"], "follow_up_query": "search for X"}'
    )
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="test")
    state["plan"] = ["step 1"]
    state["summary"] = "a summary"
    state["gap_retry_count"] = MAX_GAP_RETRIES  # already at the cap
    update = gap_finder_node(state)

    # must NOT trigger another retry -- cap reached
    assert "gap_retrieval_pending" not in update
    assert update["gaps"] == ["still missing X"]  # remaining gap surfaced, not silently dropped


@patch("app.agents.gap_finder_agent.get_llm")
def test_gap_finder_falls_back_gracefully_on_malformed_json(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json at all")
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="test")
    state["plan"] = ["step 1"]
    state["summary"] = "a summary"
    update = gap_finder_node(state)

    assert update["gaps"] == []  # assumes no gaps rather than crashing


# Full end-to-end graph execution tests (proving the bounded gap-retry
# loop actually terminates) now live in tests/test_module7_agents.py,
# since a complete graph run also passes through the Literature Review
# node added in Module 7. Keeping one authoritative full-graph test file
# avoids two files drifting out of sync with the graph's actual node set.
