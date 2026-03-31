"""Runs a fully local video-to-RAG workflow on a short sample clip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from rag_system.video_rag import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LOCAL_LLM_MODEL_NAME,
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    DEFAULT_OPENAI_LLM_MODEL_NAME,
    DEFAULT_QUERY,
    DEFAULT_SAMPLE_DURATION_SECONDS,
    DEFAULT_VIDEO_FILE_NAME,
    DEFAULT_WHISPER_MODEL_NAME,
    run_local_video_rag,
)


def main() -> None:
    """Extracts, transcribes, indexes, and queries a local video sample."""

    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    use_openai = _as_bool(os.getenv("USE_OPENAI", "false"))
    embedding_provider = os.getenv(
        "VIDEO_EMBEDDING_PROVIDER",
        "openai" if use_openai else "sentence_transformers",
    )
    embedding_model = os.getenv(
        "VIDEO_EMBEDDING_MODEL",
        DEFAULT_OPENAI_EMBEDDING_MODEL_NAME if use_openai else DEFAULT_EMBEDDING_MODEL_NAME,
    )
    llm_provider = os.getenv(
        "VIDEO_LLM_PROVIDER",
        "openai" if use_openai else "huggingface_local",
    )
    llm_model = os.getenv(
        "VIDEO_LLM_MODEL",
        DEFAULT_OPENAI_LLM_MODEL_NAME if use_openai else DEFAULT_LOCAL_LLM_MODEL_NAME,
    )

    print("[STEP] Running local video-to-RAG workflow...")
    result = run_local_video_rag(
        project_dir=project_dir,
        video_name=args.video_name,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        whisper_model=args.whisper_model,
        embedding_provider=embedding_provider,
        embedding_model=args.embedding_model or embedding_model,
        llm_provider=llm_provider,
        local_llm_model=args.local_llm_model or llm_model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        query=args.query,
        debug=True,
    )

    print(f"[INFO] Sample video: {result.sample_video_path}")
    print(f"[INFO] Sample audio: {result.sample_audio_path}")
    print(f"[INFO] Transcript text: {result.transcript_path}")
    print(f"[INFO] Transcript metadata: {result.transcript_metadata_path}")
    print("\n[TRANSCRIPT PREVIEW]\n")
    print(result.transcription_text[:1000] or "(no transcript text produced)")
    print(f"\n[INFO] Index stats: {result.index_stats}")
    print("\n[QUERY]\n")
    print(args.query)
    print("\n[ANSWER]\n")
    print(result.answer_result.answer)
    print("\n[RETRIEVED CHUNKS]\n")
    for item in result.answer_result.retrieved_results:
        print(
            {
                "score": round(item.score, 4),
                "source": item.metadata.get("source"),
                "chunk_index": item.metadata.get("chunk_index"),
            }
        )


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the local video RAG workflow."""

    parser = argparse.ArgumentParser(description="Run local video transcription and RAG.")
    parser.add_argument("--video-name", default=DEFAULT_VIDEO_FILE_NAME, help="Video file name in the project root.")
    parser.add_argument("--start-seconds", type=int, default=0, help="Start offset for the sample clip.")
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_SAMPLE_DURATION_SECONDS,
        help="Duration of the extracted sample clip.",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL_NAME, help="Local Whisper model name.")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Sentence Transformers embedding model name.",
    )
    parser.add_argument("--local-llm-model", default=None, help="Answer model name.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question to ask over the transcript.")
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    """Converts a string environment flag into a boolean."""

    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
