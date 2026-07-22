"""
Shared Gemini LLM client accessor.

Why one shared function instead of each agent instantiating its own
ChatGoogleGenerativeAI:
    Every agent needs the same model name, temperature, and API key from
    Settings. Centralizing this means: (1) switching Gemini models is a
    one-line config change, not a find-and-replace across five agent
    files, and (2) the client is cached, so we're not paying client
    construction overhead on every single agent invocation within a
    graph run.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


@lru_cache
def get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
    )
