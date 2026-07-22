"""
Citation Agent.

Design decision worth defending in an interview: this agent does NOT call
an LLM. Citations are built deterministically, directly from
retrieved_chunks metadata (which document/page or which URL each chunk
actually came from). If we instead asked an LLM to "generate citations
for this summary," it could hallucinate a source, misattribute a claim,
or cite a chunk that was never actually retrieved -- exactly the kind of
error that makes RAG citations untrustworthy. Deterministic construction
guarantees every citation traces back to a real, retrieved piece of
evidence.

Output shape: state["citations"] is a list of dicts (JSON-serializable,
since it gets persisted into Report.citations_json in Module 2's schema)
with a stable index so the final report (Module 7) can reference
citations as [1], [2], etc.
"""

from app.graph.state import AgentStep, ResearchState


def citation_node(state: ResearchState) -> dict:
    citations = []
    for i, chunk in enumerate(state["retrieved_chunks"], start=1):
        if chunk["source"] == "local_document":
            citation = {
                "index": i,
                "source_type": "local_document",
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
                "excerpt": chunk["text"][:200],
            }
        else:
            # web_search chunks embed "title: content (source: url)" --
            # pull the URL back out for a clean citation record.
            text = chunk["text"]
            url = ""
            if "(source: " in text:
                url = text.rsplit("(source: ", 1)[-1].rstrip(")")
            citation = {
                "index": i,
                "source_type": "web_search",
                "url": url,
                "excerpt": text[:200],
            }
        citations.append(citation)

    trace_entry: AgentStep = {
        "agent_name": "citation_agent",
        "summary": f"Generated {len(citations)} citation(s) from retrieved chunks",
        "data": {"num_citations": len(citations)},
    }

    return {"citations": citations, "agent_trace": [trace_entry]}
