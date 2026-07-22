"""
Tests for Module 5: Memory, Planner, Retriever sufficiency logic,
WebSearch, and the Manager's extended routing rules.

Gemini and Tavily calls are mocked here (I don't have a live API key in
this sandbox) -- these tests verify prompt construction, response
parsing, and control flow. A live pass with a real GOOGLE_API_KEY /
TAVILY_API_KEY is still needed to confirm actual model behavior; see the
README note for this module.
"""

from unittest.mock import MagicMock, patch

from app.agents.manager_agent import manager_node
from app.agents.memory_agent import memory_node
from app.agents.planner_agent import _extract_json_array, planner_node
from app.agents.retriever_agent import MIN_ACCEPTABLE_SCORE, retrieval_is_sufficient
from app.agents.websearch_agent import websearch_node
from app.graph.state import create_initial_state


# --- Memory Agent ---

def test_memory_node_with_no_prior_steps():
    state = create_initial_state(session_id=1, query="test")
    update = memory_node(state)
    assert "No prior steps" in update["memory_context"]


def test_memory_node_condenses_agent_trace():
    state = create_initial_state(session_id=1, query="test")
    state["agent_trace"] = [
        {"agent_name": "planner_agent", "summary": "Made a 3-step plan", "data": {}},
        {"agent_name": "retriever_agent", "summary": "Found 2 chunks", "data": {}},
    ]
    update = memory_node(state)
    assert "planner_agent" in update["memory_context"]
    assert "Made a 3-step plan" in update["memory_context"]
    assert "retriever_agent" in update["memory_context"]


# --- Planner Agent (mocked LLM) ---

def test_extract_json_array_handles_markdown_fences():
    raw = '```json\n["step 1", "step 2"]\n```'
    assert _extract_json_array(raw) == ["step 1", "step 2"]


def test_extract_json_array_handles_plain_json():
    raw = '["step 1", "step 2", "step 3"]'
    assert _extract_json_array(raw) == ["step 1", "step 2", "step 3"]


@patch("app.agents.planner_agent.get_llm")
def test_planner_node_parses_valid_json_response(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["Define RAG", "Find papers", "Compare approaches"]')
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="What is RAG?")
    update = planner_node(state)

    assert update["plan"] == ["Define RAG", "Find papers", "Compare approaches"]
    assert update["agent_trace"][0]["agent_name"] == "planner_agent"


@patch("app.agents.planner_agent.get_llm")
def test_planner_node_falls_back_on_malformed_json(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Sure! Here's a plan: 1. Do stuff 2. Do more stuff")
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="What is RAG?")
    update = planner_node(state)

    # should not crash -- falls back to a single-step plan
    assert len(update["plan"]) == 1
    assert "What is RAG?" in update["plan"][0]


@patch("app.agents.planner_agent.get_llm")
def test_planner_node_includes_memory_context_in_prompt(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["step"]')
    mock_get_llm.return_value = mock_llm

    state = create_initial_state(session_id=1, query="test query")
    state["memory_context"] = "Steps taken so far: retrieved 3 chunks about llamas."
    planner_node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message = call_args[1][1]
    assert "llamas" in human_message


# --- Retriever sufficiency predicate ---

def test_retrieval_is_sufficient_true_with_enough_good_chunks():
    state = create_initial_state(session_id=1, query="test")
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "a", "score": MIN_ACCEPTABLE_SCORE + 0.1, "source": "local_document"},
        {"document_id": 1, "page_number": 2, "text": "b", "score": MIN_ACCEPTABLE_SCORE + 0.2, "source": "local_document"},
    ]
    assert retrieval_is_sufficient(state) is True


def test_retrieval_is_sufficient_false_with_too_few_chunks():
    state = create_initial_state(session_id=1, query="test")
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "a", "score": 0.9, "source": "local_document"},
    ]
    assert retrieval_is_sufficient(state) is False


def test_retrieval_is_sufficient_false_with_low_scores():
    state = create_initial_state(session_id=1, query="test")
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "a", "score": 0.1, "source": "local_document"},
        {"document_id": 1, "page_number": 2, "text": "b", "score": 0.05, "source": "local_document"},
    ]
    assert retrieval_is_sufficient(state) is False


def test_retrieval_is_sufficient_ignores_web_chunks():
    """Only local_document chunks should count toward sufficiency --
    otherwise a prior web search result would mask genuinely thin local
    retrieval on a later loop."""
    state = create_initial_state(session_id=1, query="test")
    state["retrieved_chunks"] = [
        {"document_id": -1, "page_number": 0, "text": "web result", "score": 0.99, "source": "web_search"},
    ]
    assert retrieval_is_sufficient(state) is False


# --- WebSearch Agent (mocked Tavily) ---

@patch("app.agents.websearch_agent.TavilyClient")
def test_websearch_node_normalizes_results(mock_client_cls):
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "FAISS Docs", "content": "FAISS is a vector search library.", "url": "https://faiss.ai", "score": 0.8},
        ]
    }
    mock_client_cls.return_value = mock_client

    state = create_initial_state(session_id=1, query="what is faiss")
    update = websearch_node(state)

    assert update["web_search_attempted"] is True
    assert len(update["retrieved_chunks"]) == 1
    assert update["retrieved_chunks"][0]["source"] == "web_search"
    assert "FAISS" in update["retrieved_chunks"][0]["text"]


@patch("app.agents.websearch_agent.TavilyClient")
def test_websearch_node_handles_api_failure_gracefully(mock_client_cls):
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Tavily API down")
    mock_client_cls.return_value = mock_client

    state = create_initial_state(session_id=1, query="test")
    update = websearch_node(state)

    # should not raise -- graph must continue even if web search fails
    assert update["web_search_attempted"] is True
    assert "agent_trace" in update


# --- Manager routing (Module 5 rules) ---

def test_manager_routes_to_memory_first():
    state = create_initial_state(session_id=1, query="test")
    update = manager_node(state)
    assert update["next_agent"] == "memory"


def test_manager_routes_to_planner_after_memory():
    state = create_initial_state(session_id=1, query="test")
    state["memory_context"] = "some context"
    update = manager_node(state)
    assert update["next_agent"] == "planner"


def test_manager_routes_to_websearch_when_retrieval_thin():
    state = create_initial_state(session_id=1, query="test")
    state["memory_context"] = "ctx"
    state["plan"] = ["step"]
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "a", "score": 0.1, "source": "local_document"},
    ]
    update = manager_node(state)
    assert update["next_agent"] == "websearch"


def test_manager_skips_websearch_when_already_attempted():
    state = create_initial_state(session_id=1, query="test")
    state["memory_context"] = "ctx"
    state["plan"] = ["step"]
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "a", "score": 0.1, "source": "local_document"},
    ]
    state["web_search_attempted"] = True
    update = manager_node(state)
    assert update["next_agent"] == "summarizer"


def test_manager_skips_websearch_when_retrieval_already_good():
    state = create_initial_state(session_id=1, query="test")
    state["memory_context"] = "ctx"
    state["plan"] = ["step"]
    state["retrieved_chunks"] = [
        {"document_id": 1, "page_number": 1, "text": "a", "score": 0.9, "source": "local_document"},
        {"document_id": 1, "page_number": 2, "text": "b", "score": 0.8, "source": "local_document"},
    ]
    update = manager_node(state)
    assert update["next_agent"] == "summarizer"
