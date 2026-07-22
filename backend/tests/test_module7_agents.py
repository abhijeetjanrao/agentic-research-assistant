"""
Tests for Module 7: Literature Review Agent and Report Generator Agent,
including real MySQL persistence and full graph runs.
"""

import json
from unittest.mock import MagicMock, patch

from app.agents.literature_review_agent import literature_review_node
from app.agents.report_agent import _format_citations_section, _format_limitations_section, report_generator_node
from app.graph.state import create_initial_state


# --- Literature Review Agent (mocked LLM) ---

@patch("app.agents.literature_review_agent.get_llm")
def test_literature_review_node_calls_llm_with_summary_and_comparison(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="## Background\n...\n## Key Findings\n...")
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="RAG vs fine-tuning")
    state["summary"] = "RAG combines retrieval and generation."
    state["comparison"] = "RAG is cheaper to update; fine-tuning bakes knowledge into weights."
    update = literature_review_node(state)

    assert "Background" in update["literature_review"]
    prompt_text = mock_llm.invoke.call_args[0][0][1][1]
    assert "fine-tuning bakes knowledge" in prompt_text


@patch("app.agents.literature_review_agent.get_llm")
def test_literature_review_node_omits_comparison_when_none(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="## Background\n...")
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="What is RAG?")
    state["summary"] = "RAG combines retrieval and generation."
    state["comparison"] = None
    literature_review_node(state)

    prompt_text = mock_llm.invoke.call_args[0][0][1][1]
    assert "Comparison:" not in prompt_text


# --- Report Generator formatting helpers ---

def test_format_citations_section_empty():
    assert "No citations" in _format_citations_section([])


def test_format_citations_section_mixed_sources():
    citations = [
        {"index": 1, "source_type": "local_document", "document_id": 3, "page_number": 2, "excerpt": "some text"},
        {"index": 2, "source_type": "web_search", "url": "https://example.com", "excerpt": "web text"},
    ]
    formatted = _format_citations_section(citations)
    assert "[1]" in formatted and "page 2" in formatted
    assert "[2]" in formatted and "https://example.com" in formatted


def test_format_limitations_section_no_gaps():
    assert "No significant gaps" in _format_limitations_section([])


def test_format_limitations_section_with_gaps():
    formatted = _format_limitations_section(["missing cost analysis", "no recent 2026 sources"])
    assert "missing cost analysis" in formatted
    assert "no recent 2026 sources" in formatted


# --- Report Generator Agent: real MySQL persistence ---

@patch("app.agents.report_agent.SessionLocal")
def test_report_generator_persists_to_db_and_returns_report(mock_session_local):
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    state = create_initial_state(session_id=1, query="What is RAG?")
    state["literature_review"] = "## Background\nRAG is a technique..."
    state["citations"] = [{"index": 1, "source_type": "local_document", "document_id": 1, "page_number": 1, "excerpt": "x"}]
    state["gaps"] = []

    update = report_generator_node(state)

    assert "# Research Report: What is RAG?" in update["final_report"]
    assert "RAG is a technique" in update["final_report"]
    assert "## Citations" in update["final_report"]
    assert "## Limitations" in update["final_report"]
    assert update["agent_trace"][0]["data"]["db_saved"] is True
    mock_db.close.assert_called_once()


@patch("app.agents.report_agent.SessionLocal")
def test_report_generator_survives_db_failure(mock_session_local):
    """A DB outage must not destroy the finished report sitting in memory
    -- the graph should still return final_report even if persistence fails."""
    mock_session_local.side_effect = Exception("MySQL connection refused")

    state = create_initial_state(session_id=1, query="test")
    state["literature_review"] = "content"
    state["citations"] = []
    state["gaps"] = []

    update = report_generator_node(state)

    assert update["final_report"] is not None
    assert "content" in update["final_report"]
    assert update["agent_trace"][0]["data"]["db_saved"] is False


def test_report_generator_persists_to_real_mysql(real_mysql_db_session):
    """End-to-end proof against the actual MySQL instance (not a mock):
    the report row really exists afterward, with citations serialized as
    valid JSON."""
    from app.db import crud

    session = crud.create_session(real_mysql_db_session, title="Module 7 real DB test")

    state = create_initial_state(session_id=session.id, query="What is RAG?")
    state["literature_review"] = "## Background\nReal DB test content."
    state["citations"] = [{"index": 1, "source_type": "local_document", "document_id": 1, "page_number": 1, "excerpt": "x"}]
    state["gaps"] = ["a remaining gap"]

    with patch("app.agents.report_agent.SessionLocal", return_value=real_mysql_db_session):
        # SessionLocal() is called inside report_generator_node; patch it
        # to return our real, already-open test session instead of opening
        # a new one, so we can inspect the same transaction afterward.
        # (report_generator_node closes it -- fine, this test's fixture
        # will have already committed by then.)
        original_close = real_mysql_db_session.close
        real_mysql_db_session.close = lambda: None  # prevent early close so we can query after
        try:
            report_generator_node(state)
        finally:
            real_mysql_db_session.close = original_close

    reports = crud.get_session_reports(real_mysql_db_session, session.id)
    assert len(reports) == 1
    assert "Real DB test content" in reports[0].content_markdown
    parsed_citations = json.loads(reports[0].citations_json)
    assert parsed_citations[0]["document_id"] == 1


# --- Full graph: the complete pipeline, every boundary mocked ---

@patch("app.agents.report_agent.SessionLocal")
@patch("app.agents.websearch_agent.TavilyClient")
@patch("app.agents.retriever_agent.get_vector_store")
@patch("app.agents.retriever_agent.get_embedding_model")
@patch("app.agents.literature_review_agent.get_llm")
@patch("app.agents.gap_finder_agent.get_llm")
@patch("app.agents.compare_agent.get_llm")
@patch("app.agents.summary_agent.get_llm")
@patch("app.agents.planner_agent.get_llm")
def test_full_graph_happy_path_produces_persisted_report(
    mock_planner_llm,
    mock_summary_llm,
    mock_compare_llm,
    mock_gapfinder_llm,
    mock_litreview_llm,
    mock_get_embedding_model,
    mock_get_vector_store,
    mock_tavily_cls,
    mock_session_local,
):
    """The complete pipeline, happy path: strong local retrieval, no
    gaps, straight through to a persisted final report. This is the
    single test that exercises every node in the graph in one pass."""
    from app.graph.research_graph import build_research_graph

    mock_planner_llm.return_value.invoke.return_value = MagicMock(
        content='["Define RAG", "Summarize evidence"]'
    )
    mock_summary_llm.return_value.invoke.return_value = MagicMock(content="RAG combines retrieval and generation.")
    mock_compare_llm.return_value.invoke.return_value = MagicMock(content="n/a")
    mock_gapfinder_llm.return_value.invoke.return_value = MagicMock(
        content='{"gaps": [], "follow_up_query": null}'
    )
    mock_litreview_llm.return_value.invoke.return_value = MagicMock(
        content="## Background\nRAG background.\n## Key Findings\nRAG findings."
    )

    mock_get_embedding_model.return_value.embed_query.return_value = [0.1] * 384
    good_result = MagicMock()
    good_result.metadata.document_id = 1
    good_result.metadata.page_number = 1
    good_result.metadata.text = "Strong local content about RAG."
    good_result.score = 0.9
    mock_get_vector_store.return_value.search.return_value = [good_result, good_result]

    mock_db = MagicMock()
    mock_session_local.return_value = mock_db

    graph = build_research_graph()
    initial_state = create_initial_state(session_id=1, query="What is retrieval-augmented generation?")
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

    # web search must never fire -- local retrieval was strong
    mock_tavily_cls.assert_not_called()

    assert final_state["plan"] is not None
    assert final_state["summary"] is not None
    assert final_state["citations"] is not None
    assert final_state["gaps"] == []
    assert final_state["literature_review"] is not None
    assert "RAG background" in final_state["literature_review"]
    assert final_state["final_report"] is not None
    assert "# Research Report:" in final_state["final_report"]
    assert "## Citations" in final_state["final_report"]
    assert "## Limitations" in final_state["final_report"]

    # every agent should appear exactly once in the happy path
    agent_names = [step["agent_name"] for step in final_state["agent_trace"]]
    for expected in [
        "memory_agent", "planner_agent", "retriever_agent", "summary_agent",
        "compare_agent", "citation_agent", "gap_finder_agent",
        "literature_review_agent", "report_generator_agent",
    ]:
        assert expected in agent_names, f"{expected} did not run"
    assert "websearch_agent" not in agent_names

    # the report generator should have attempted to persist to MySQL
    mock_db.close.assert_called()
