"""
Research graph: wires the Manager Agent and worker agent nodes into a
LangGraph StateGraph.

Why every worker routes back to the manager (a "star" topology) instead
of a straight-line chain:
    A linear chain hardcodes the order forever. Routing every worker back
    through the manager means the *manager* decides the next step each
    time, based on current state -- which is what allows the Gap Finder
    (Module 6) to send control back to the Retriever instead of forward
    to the Report Generator, without changing this graph's structure at
    all. Only manager_agent.py's routing logic needs to change for that.

This is the actual mechanism that makes the system "agentic" rather than
a fixed pipeline: the path through the graph is decided at runtime.
"""

from langgraph.graph import END, StateGraph

from app.agents.compare_agent import compare_node
from app.agents.citation_agent import citation_node
from app.agents.gap_finder_agent import gap_finder_node
from app.agents.literature_review_agent import literature_review_node
from app.agents.manager_agent import manager_node, route_after_manager
from app.agents.memory_agent import memory_node
from app.agents.planner_agent import planner_node
from app.agents.report_agent import report_generator_node
from app.agents.retriever_agent import retriever_node
from app.agents.summary_agent import summarizer_node
from app.agents.websearch_agent import websearch_node
from app.graph.state import ResearchState


def build_research_graph():
    """Construct and compile the graph. Called once at app startup
    (Module 8 will cache the compiled graph the same way we cache the
    embedding model and vector store) rather than rebuilt per request."""
    graph = StateGraph(ResearchState)

    graph.add_node("manager", manager_node)
    graph.add_node("memory", memory_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("websearch", websearch_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("compare", compare_node)
    graph.add_node("citation", citation_node)
    graph.add_node("gap_finder", gap_finder_node)
    graph.add_node("lit_review", literature_review_node)
    graph.add_node("report_generator", report_generator_node)

    graph.set_entry_point("manager")

    # Every worker node, after running, returns control to the manager --
    # the manager then decides the next hop based on updated state.
    graph.add_conditional_edges(
        "manager",
        route_after_manager,
        {
            "memory": "memory",
            "planner": "planner",
            "retriever": "retriever",
            "websearch": "websearch",
            "summarizer": "summarizer",
            "compare": "compare",
            "citation": "citation",
            "gap_finder": "gap_finder",
            "lit_review": "lit_review",
            "report_generator": "report_generator",
            "end": END,
        },
    )
    graph.add_edge("memory", "manager")
    graph.add_edge("planner", "manager")
    graph.add_edge("retriever", "manager")
    graph.add_edge("websearch", "manager")
    graph.add_edge("summarizer", "manager")
    graph.add_edge("compare", "manager")
    graph.add_edge("citation", "manager")
    graph.add_edge("gap_finder", "manager")
    graph.add_edge("lit_review", "manager")
    graph.add_edge("report_generator", "manager")

    return graph.compile()


_compiled_graph = None


def get_research_graph():
    """Process-wide singleton for the compiled graph, mirroring the
    pattern used for the embedding model and vector store singletons."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_research_graph()
    return _compiled_graph
