"""
Summary Agent.

Replaces the Module 4 stub with a real Gemini call that synthesizes all
retrieved chunks (local documents + web search results, if any) into one
coherent narrative summary.

Prompt engineering decision: chunks are presented to the model labeled
with their source (local document page N, or a web URL) so the summary
can implicitly ground claims in a traceable origin -- the Citation Agent
(next in this module) then builds the actual citation list from the same
retrieved_chunks metadata, so summary text and citations always refer to
the same underlying evidence.
"""

from app.agents.llm import get_llm
from app.graph.state import AgentStep, ResearchState

SUMMARY_SYSTEM_PROMPT = """You are the Summary Agent in a multi-agent research assistant.
You will be given a research query and a set of retrieved text chunks (from local
documents and/or web search). Write a clear, well-organized summary (3-6 paragraphs)
that synthesizes the key information relevant to the query.

Do not fabricate information not present in the provided chunks. If the chunks are
sparse or only partially relevant, summarize what is genuinely there rather than
padding with general knowledge.
"""


def _format_chunks(state: ResearchState) -> str:
    lines = []
    for i, chunk in enumerate(state["retrieved_chunks"], start=1):
        origin = (
            f"local document #{chunk['document_id']}, page {chunk['page_number']}"
            if chunk["source"] == "local_document"
            else "web search result"
        )
        lines.append(f"[Chunk {i} - {origin}]\n{chunk['text']}")
    return "\n\n".join(lines) if lines else "(no chunks retrieved)"


def summarizer_node(state: ResearchState) -> dict:
    llm = get_llm()
    user_prompt = (
        f"Research query: {state['query']}\n\n"
        f"Retrieved chunks:\n{_format_chunks(state)}\n\n"
        f"Write the summary now."
    )

    response = llm.invoke(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )
    summary = response.content.strip()

    trace_entry: AgentStep = {
        "agent_name": "summary_agent",
        "summary": f"Summarized {len(state['retrieved_chunks'])} chunk(s)",
        "data": {"num_chunks": len(state["retrieved_chunks"])},
    }

    return {"summary": summary, "agent_trace": [trace_entry]}
