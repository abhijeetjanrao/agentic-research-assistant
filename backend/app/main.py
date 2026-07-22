"""
FastAPI application entrypoint.

At this stage (Module 1) this only wires up config + logging + a health
check, so we can verify the foundation works end-to-end before adding DB,
RAG, or agents in later modules. Routers for chat/documents/reports get
included here once they exist (Module 8).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import setup_logging, logger

settings = get_settings()
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Using lifespan (not the deprecated on_event)
    keeps this future-proof and gives us a single place to init/teardown
    resources like DB connection pools once Module 2 adds the database."""
    logger.info(
        "Starting {app_name} in {env} mode (debug={debug})",
        app_name=settings.app_name,
        env=settings.app_env,
        debug=settings.debug,
    )
    yield
    logger.info("Shutting down {app_name}", app_name=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Basic liveness probe -- also useful for AWS load balancer health checks later."""
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


# Routers
from app.api import routes_chat, routes_documents, routes_reports

app.include_router(routes_chat.router, prefix=settings.api_v1_prefix)
app.include_router(routes_documents.router, prefix=settings.api_v1_prefix)
app.include_router(routes_reports.router, prefix=settings.api_v1_prefix)
