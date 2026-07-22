"""
Streamlit frontend for the Agentic Research Assistant.

Design decisions:
    - This is a thin client: all logic (agents, RAG, DB) lives in the
      FastAPI backend. Streamlit only calls the REST API and renders
      responses. This matters for the "why FastAPI + Streamlit, not just
      Streamlit alone" story: the backend is independently useful/
      testable/deployable, and swapping to a React frontend later would
      require zero backend changes.
    - Session state (st.session_state) holds only UI-local state (which
      backend session_id is active, chat history for display) -- the
      actual source of truth for a research session lives in MySQL via
      the backend, not in Streamlit's session state. Refreshing the page
      and re-selecting a session_id would recover history via the API,
      not from browser memory.
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎", layout="wide")


def api_get(path: str, **kwargs):
    response = requests.get(f"{API_BASE}{path}", timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs):
    response = requests.post(f"{API_BASE}{path}", timeout=300, **kwargs)
    response.raise_for_status()
    return response.json()


# --- Session state init ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

st.title("🔎 Agentic Research Assistant")
st.caption("Multi-agent research: plan → retrieve → synthesize → cite → report")

# --- Sidebar: session management ---
with st.sidebar:
    st.header("Research Session")

    try:
        sessions = api_get("/sessions")
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot reach backend API: {e}")
        sessions = []

    session_options = {f"#{s['id']} — {s['title']}": s["id"] for s in sessions}

    new_title = st.text_input("New session title", placeholder="e.g. RAG vs fine-tuning")
    if st.button("➕ Create session", use_container_width=True) and new_title.strip():
        try:
            created = api_post("/sessions", json={"title": new_title.strip()})
            st.session_state.session_id = created["id"]
            st.session_state.chat_history = []
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to create session: {e}")

    if session_options:
        st.divider()
        selected_label = st.selectbox(
            "Or pick an existing session",
            options=list(session_options.keys()),
            index=None,
            placeholder="Select a session...",
        )
        if selected_label:
            st.session_state.session_id = session_options[selected_label]

    st.divider()

    if st.session_state.session_id:
        st.subheader("📄 Upload documents")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file and st.button("Ingest document", use_container_width=True):
            with st.spinner("Ingesting PDF (extracting, chunking, embedding)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    doc = requests.post(
                        f"{API_BASE}/sessions/{st.session_state.session_id}/documents",
                        files=files,
                        timeout=300,
                    ).json()
                    if doc.get("status") == "ingested":
                        st.success(f"Ingested {doc['filename']} ({doc['num_chunks']} chunks)")
                    else:
                        st.error(f"Ingestion failed: {doc.get('error_message')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: {e}")

        st.divider()
        try:
            docs = api_get(f"/sessions/{st.session_state.session_id}/documents")
            if docs:
                st.subheader("Documents in this session")
                for d in docs:
                    icon = {"ingested": "✅", "failed": "❌", "ingesting": "⏳"}.get(d["status"], "•")
                    st.caption(f"{icon} {d['filename']} ({d['status']})")
        except requests.exceptions.RequestException:
            pass


# --- Main panel: chat / research ---
if not st.session_state.session_id:
    st.info("Create or select a session in the sidebar to begin.")
else:
    st.subheader(f"Session #{st.session_state.session_id}")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a research question...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Running multi-agent research (plan → retrieve → synthesize → cite)..."):
                try:
                    result = api_post(
                        "/research",
                        json={"session_id": st.session_state.session_id, "query": query},
                    )
                    report = result.get("final_report") or "_No report was generated._"
                    st.markdown(report)

                    if result.get("gaps"):
                        with st.expander("⚠️ Remaining research gaps"):
                            for gap in result["gaps"]:
                                st.markdown(f"- {gap}")

                    with st.expander("🧠 Agent trace (what each agent did)"):
                        for step in result.get("agent_trace", []):
                            st.markdown(f"**{step['agent_name']}**: {step['summary']}")

                    st.session_state.chat_history.append({"role": "assistant", "content": report})
                except requests.exceptions.RequestException as e:
                    error_msg = f"Research request failed: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
