"""
WebSearch Agent.

When this runs (per the Manager's routing logic, not this file):
    Only when the Retriever's local FAISS results are insufficient (see
    retriever_agent.retrieval_is_sufficient). This keeps web search as a
    fallback for gaps in the user's uploaded documents, rather than
    always hitting an external API on every query -- cheaper, faster, and
    arguably more correct: if the user uploaded the relevant paper, we
    should trust and cite that over a live web result.

Why Tavily specifically:
    It's built for LLM/agent consumption (returns clean extracted content
    with a relevance score per result, not raw HTML/SERP data), has
    native LangChain support, and a workable free tier for a portfolio
    project -- so no custom HTML scraping/parsing code is needed here.

Output shape: web results are normalized into the same RetrievedChunk
TypedDict the Retriever Agent produces (with source="web_search"), so
every downstream synthesis agent treats local and web-sourced content
identically instead of needing source-specific branches everywhere.
"""

from tavily import TavilyClient

from app.config import get_settings
from app.graph.state import AgentStep, RetrievedChunk, ResearchState
from app.logging_config import logger

MAX_WEB_RESULTS = 5


def websearch_node(state: ResearchState) -> dict:
    settings = get_settings()
    client = TavilyClient(api_key=settings.tavily_api_key)
    search_query = state.get("gap_query") or state["query"]

    try:
        response = client.search(
            query=search_query, max_results=MAX_WEB_RESULTS, include_raw_content=False
        )
    except Exception as e:
        # A web search failure shouldn't crash the whole graph run --
        # local retrieval results (if any) still flow through, and the
        # report generator can note the gap.
        logger.error("WebSearch agent failed: {error}", error=str(e))
        trace_entry: AgentStep = {
            "agent_name": "websearch_agent",
            "summary": f"Web search failed: {e}",
            "data": {"error": str(e)},
        }
        return {"web_search_attempted": True, "agent_trace": [trace_entry]}

    chunks: list[RetrievedChunk] = [
        {
            "document_id": -1,  # no local Document row; -1 signals "external"
            "page_number": 0,
            "text": f"{result.get('title', '')}: {result.get('content', '')} (source: {result.get('url', '')})",
            "score": result.get("score", 0.0),
            "source": "web_search",
        }
        for result in response.get("results", [])
    ]

    trace_entry: AgentStep = {
        "agent_name": "websearch_agent",
        "summary": f"Found {len(chunks)} web result(s) (local retrieval was insufficient)",
        "data": {"num_results": len(chunks)},
    }

    return {
        "retrieved_chunks": chunks,
        "web_search_attempted": True,
        "agent_trace": [trace_entry],
    }
