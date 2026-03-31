"""Simple interactive CLI for dynamic RAG queries."""

from __future__ import annotations

import os
from pathlib import Path
import sys
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_SITE_PACKAGES = PROJECT_DIR / "vendor_pkgs"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

from rag_system import RAGPipeline, build_default_config


def main() -> None:
    """Builds or loads the index and starts a small query loop."""

    if load_dotenv is not None:
        load_dotenv()

    project_dir = PROJECT_DIR
    config = build_default_config(project_dir)
    config.collection_name = "sample_docs"
    config.embeddings.provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    config.embeddings.model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    config.embeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.embeddings.openai_base_url = os.getenv("OPENAI_BASE_URL")
    config.llm.provider = os.getenv("LLM_PROVIDER", "extractive")
    config.llm.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    config.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.llm.openai_base_url = os.getenv("OPENAI_BASE_URL")

    pipeline = RAGPipeline(config)

    try:
        pipeline.load_index()
    except Exception:
        pipeline.build_index([project_dir / "data" / "sample"])

    print("Type a question, or 'exit' to stop.")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        result = pipeline.answer_query(query=query)
        print("\nAnswer:\n")
        print(result.answer)


if __name__ == "__main__":
    main()
