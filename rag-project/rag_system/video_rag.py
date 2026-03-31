"""Reusable helpers for the local video-to-RAG workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .models import AnswerResult
from .pipeline import RAGPipeline
from .transcription import extract_video_sample, save_transcript_artifacts, transcribe_audio
from .utils import ensure_directory
from . import build_default_config


DEFAULT_VIDEO_FILE_NAME = "video_2026-03-30_15-54-25.mp4"
DEFAULT_SAMPLE_DURATION_SECONDS = 60
DEFAULT_WHISPER_MODEL_NAME = "openai/whisper-tiny.en"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_LLM_MODEL_NAME = "google/flan-t5-small"
DEFAULT_OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DEFAULT_OPENAI_LLM_MODEL_NAME = "gpt-4o-mini"
DEFAULT_QUERY = "What is the speaker trying to teach in this video sample?"


@dataclass(slots=True)
class VideoRAGResult:
    """Stores the outputs of a local video-to-RAG run."""

    sample_video_path: Path
    sample_audio_path: Path
    transcript_path: Path
    transcript_metadata_path: Path
    transcription_text: str
    index_stats: dict[str, int]
    answer_result: AnswerResult
    collection_name: str


def build_local_video_rag_config(
    project_dir: str | Path,
    collection_name: str,
    embedding_provider: str,
    embedding_model: str,
    llm_provider: str,
    local_llm_model: str,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    debug: bool = True,
) -> PipelineConfig:
    """Builds the configuration for the local video RAG workflow."""

    config = build_default_config(project_dir)
    config.collection_name = collection_name
    config.debug = debug
    config.chunking.chunk_size = 260
    config.chunking.chunk_overlap = 50
    config.retrieval.top_k = 3
    config.retrieval.min_score = 0.15
    config.embeddings.provider = embedding_provider
    config.embeddings.model_name = embedding_model
    config.embeddings.openai_api_key = openai_api_key
    config.embeddings.openai_base_url = openai_base_url
    config.llm.provider = llm_provider
    config.llm.model_name = local_llm_model
    config.llm.hf_task = "text2text-generation"
    config.llm.max_new_tokens = 160
    config.llm.temperature = 0.0
    config.llm.openai_api_key = openai_api_key
    config.llm.openai_base_url = openai_base_url
    return config


def run_local_video_rag(
    project_dir: str | Path,
    video_name: str = DEFAULT_VIDEO_FILE_NAME,
    start_seconds: int = 0,
    duration_seconds: int = DEFAULT_SAMPLE_DURATION_SECONDS,
    whisper_model: str = DEFAULT_WHISPER_MODEL_NAME,
    embedding_provider: str = "sentence_transformers",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_NAME,
    llm_provider: str = "huggingface_local",
    local_llm_model: str = DEFAULT_LOCAL_LLM_MODEL_NAME,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    query: str = DEFAULT_QUERY,
    debug: bool = True,
) -> VideoRAGResult:
    """Runs extraction, transcription, indexing, and local RAG answering."""

    project_path = Path(project_dir).resolve()
    video_path = project_path / video_name

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    processing_dir = ensure_directory(project_path / "artifacts" / "video_processing")
    transcript_dir = ensure_directory(project_path / "data" / "video_transcripts")

    sample_video_path, sample_audio_path = extract_video_sample(
        video_path=video_path,
        output_dir=processing_dir,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )

    transcription = transcribe_audio(
        audio_path=sample_audio_path,
        model_name=whisper_model,
        chunk_length_s=20,
    )
    transcript_stem = f"{video_path.stem}_sample_{duration_seconds}s_transcript"
    transcript_path, metadata_path = save_transcript_artifacts(
        transcription=transcription,
        output_dir=transcript_dir,
        stem=transcript_stem,
    )

    collection_name = "local_video_sample"
    pipeline = RAGPipeline(
        config=build_local_video_rag_config(
            project_dir=project_path,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            local_llm_model=local_llm_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            debug=debug,
        )
    )
    index_stats = pipeline.build_index([transcript_path])
    answer_result = pipeline.answer_query(query=query, top_k=3, min_score=0.15)

    return VideoRAGResult(
        sample_video_path=sample_video_path,
        sample_audio_path=sample_audio_path,
        transcript_path=transcript_path,
        transcript_metadata_path=metadata_path,
        transcription_text=str(transcription["text"]).strip(),
        index_stats=index_stats,
        answer_result=answer_result,
        collection_name=collection_name,
    )
