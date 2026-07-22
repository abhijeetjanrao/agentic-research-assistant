"""
Retriever Agent.

Replaces the Module 4 stub with a real FAISS similarity search against
this session's ingested documents (via app.rag.vector_store).

Design decision: minimum similarity threshold + minimum chunk count are
exposed as module constants (not buried magic numbers) because the
Manager Agent's routing logic needs to reference the same threshold to
decide whether local retrieval was "good enough" or whether WebSearch
should run -- see manager_agent.py's route_after_manager. Keeping the
threshold in one place (imported by both) avoids the two agents silently
disagreeing about what counts as "enough."
"""

from app.rag.embeddings import get_embedding_model
from app.graph.state import AgentStep, RetrievedChunk, ResearchState
from app.logging_config import logger
from app.rag.vector_store import get_vector_store

TOP_K = 5
MIN_ACCEPTABLE_CHUNKS = 2
MIN_ACCEPTABLE_SCORE = 0.35  # cosine similarity (since embeddings are normalized)


def retriever_node(state: ResearchState) -> dict:
    # If the Gap Finder triggered a follow-up retrieval, search using its
    # targeted query instead of the original one -- that's the whole
    # point of the loop (a smarter second attempt, not a repeat search).
    search_query = state.get("gap_query") or state["query"]
    query_vector = get_embedding_model().embed_query(search_query)
    store = get_vector_store()

    results = store.search(query_vector, top_k=TOP_K, session_id=state["session_id"])

    chunks: list[RetrievedChunk] = [
        {
            "document_id": r.metadata.document_id,
            "page_number": r.metadata.page_number,
            "text": r.metadata.text,
            "score": r.score,
            "source": "local_document",
        }
        for r in results
    ]

    logger.info(
        "Retriever found {n} local chunks for session {sid} (query: {q})",
        n=len(chunks),
        sid=state["session_id"],
        q=search_query,
    )

    trace_entry: AgentStep = {
        "agent_name": "retriever_agent",
        "summary": f"Retrieved {len(chunks)} chunk(s) from local documents (query: '{search_query}')",
        "data": {
            "num_chunks": len(chunks),
            "best_score": max((c["score"] for c in chunks), default=0.0),
        },
    }

    # Clear the routing trigger flag (not the query text itself -- the
    # WebSearch agent, which may run next in this same manager loop,
    # still needs gap_query). Without this, the Manager would route back
    # to the Retriever forever, since retrieved_chunks is never empty
    # once populated.
    return {"retrieved_chunks": chunks, "gap_retrieval_pending": False, "agent_trace": [trace_entry]}


def retrieval_is_sufficient(state: ResearchState) -> bool:
    """Shared predicate used by the Manager to decide whether WebSearch
    is needed. Exposed as a function (not inlined in the manager) so it's
    independently unit-testable and so the Retriever and Manager can
    never disagree about the definition of 'sufficient.'"""
    local_chunks = [c for c in state["retrieved_chunks"] if c["source"] == "local_document"]
    if len(local_chunks) < MIN_ACCEPTABLE_CHUNKS:
        return False
    best_score = max((c["score"] for c in local_chunks), default=0.0)
    return best_score >= MIN_ACCEPTABLE_SCORE
