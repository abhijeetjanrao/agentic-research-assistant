# Agentic AI Research Assistant

A foundation for an AI research assistant that combines:
- a FastAPI backend,
- Streamlit frontend,
- PDF upload and ingestion,
- session-based research workflows,
- multi-agent orchestration via LangGraph,
- vector search with FAISS.

This repository is currently at an early build stage, with the core app
skeleton, configuration, logging, and API wiring in place.

## What’s included

- `backend/`: FastAPI service with configuration, logging, DB models,
  RAG ingestion stubs, agents, and API routes.
- `frontend/`: Streamlit app for creating sessions, uploading PDFs, and
  submitting research queries.
- `docs/`: deployment notes and architecture documentation.
- `README.md`: repository overview and local setup.

## Key features

- `backend/app/main.py`: FastAPI application entrypoint with CORS and
  health check.
- `backend/app/config.py`: centralized, validated settings loaded from
  `.env`.
- `backend/app/api/`: routes for sessions, documents, and reports.
- `backend/app/db/`: SQLAlchemy models and CRUD helpers.
- `backend/app/rag/`: embedding and ingestion utilities.
- `frontend/streamlit_app.py`: UI for session creation, PDF upload, and
  research queries.

## Run locally

1. Open a terminal in the repository root.
2. Create and activate a Python virtual environment:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` from the example and fill in required values:

```bash
copy .env.example .env
```

5. Start the backend:

```bash
uvicorn app.main:app --reload
```

6. Run the frontend in a second terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

7. Use the Streamlit UI or visit `http://localhost:8000/health` to verify
   the backend is running.

## Notes before publishing to GitHub

These files and folders should not be committed:

- `backend/venv/`
- `backend/data/`
- `backend/logs/`
- `.env`
- `.pytest_cache/`

A `.gitignore` has been added to exclude local environment files,
generated data, logs, and secrets.

## Clean GitHub upload steps

1. If you already initialized git:

```bash
git init
```

2. Add files and commit:

```bash
git add .
git commit -m "Initial clean project import"
```

3. Create a GitHub repository and connect it:

```bash
git remote add origin https://github.com/<your-user>/<repo>.git
```

4. Push to GitHub:

```bash
git branch -M main
git push -u origin main
```

## Future cleanup suggestions

- Remove `backend/venv/` if you want a smaller local repository.
- Keep `backend/data/` and `backend/logs/` out of version control.
- Add `backend/requirements.txt` if you want a separate backend-only
  dependency manifest.
- Add a short project description in the GitHub repo summary.
