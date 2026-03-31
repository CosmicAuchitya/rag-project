"""Optional FastAPI app for serving the RAG pipeline over HTTP."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
import sys
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_DIR / "vendor_pkgs"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if load_dotenv is not None:
    load_dotenv()

from rag_system import RAGPipeline, build_default_config


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError("API support requires 'fastapi' and 'pydantic'.") from exc


PIPELINE = None


class QueryRequest(BaseModel):
    """Represents one API query request."""

    query: str = Field(..., min_length=1)
    top_k: int | None = None
    min_score: float | None = None


class QueryResponse(BaseModel):
    """Represents one API query response."""

    answer: str
    sources: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the persisted index when the API starts."""

    del app
    global PIPELINE

    project_dir = os.getenv("RAG_PROJECT_DIR", os.getcwd())
    config = build_default_config(project_dir)
    config.embeddings.provider = os.getenv("EMBEDDING_PROVIDER", config.embeddings.provider)
    config.embeddings.model_name = os.getenv("EMBEDDING_MODEL", config.embeddings.model_name)
    config.embeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.embeddings.openai_base_url = os.getenv("OPENAI_BASE_URL")
    config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
    config.llm.model_name = os.getenv("LLM_MODEL", config.llm.model_name)
    config.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.llm.openai_base_url = os.getenv("OPENAI_BASE_URL")
    PIPELINE = RAGPipeline(config)

    try:
        PIPELINE.load_index()
    except Exception:
        PIPELINE = None

    yield


app = FastAPI(title="Production RAG API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Returns a simple health response."""

    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """Answers a question using the loaded RAG pipeline."""

    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="The RAG pipeline is not loaded.")

    result = PIPELINE.answer_query(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )
    sources = [str(item.metadata.get("source", "unknown")) for item in result.retrieved_results]
    return QueryResponse(answer=result.answer, sources=sources)
