"""Configuration objects for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.25
DEFAULT_RANDOM_SEED = 42
DEFAULT_CACHE_DB_NAME = "embedding_cache.sqlite"
DEFAULT_INDEX_DIR_NAME = "faiss_index"
DEFAULT_EMBEDDING_PROVIDER = "sentence_transformers"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_PROVIDER = "extractive"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_COLLECTION_NAME = "default_collection"
DEFAULT_HF_TASK = "text2text-generation"
DEFAULT_MAX_NEW_TOKENS = 192


@dataclass(slots=True)
class PathConfig:
    """Stores all filesystem paths used by the pipeline."""

    base_dir: Path
    data_dir: Path
    artifact_dir: Path
    cache_db_path: Path
    index_dir: Path


@dataclass(slots=True)
class ChunkingConfig:
    """Stores chunking parameters."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(slots=True)
class EmbeddingConfig:
    """Stores embedding configuration."""

    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    openai_api_key: str | None = None
    openai_base_url: str | None = None


@dataclass(slots=True)
class RetrievalConfig:
    """Stores retrieval parameters."""

    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE


@dataclass(slots=True)
class LLMConfig:
    """Stores LLM generation parameters."""

    provider: str = DEFAULT_LLM_PROVIDER
    model_name: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0
    max_context_results: int = 4
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    hf_task: str = DEFAULT_HF_TASK
    device: int = -1
    openai_api_key: str | None = None
    openai_base_url: str | None = None


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the entire RAG pipeline."""

    paths: PathConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    collection_name: str = DEFAULT_COLLECTION_NAME
    debug: bool = True
    random_seed: int = DEFAULT_RANDOM_SEED


def build_default_config(base_dir: str | Path) -> PipelineConfig:
    """Builds a default configuration relative to the provided base directory."""

    base_path = Path(base_dir).resolve()
    data_dir = base_path / "data"
    artifact_dir = base_path / "artifacts"
    cache_db_path = artifact_dir / DEFAULT_CACHE_DB_NAME
    index_dir = artifact_dir / DEFAULT_INDEX_DIR_NAME

    return PipelineConfig(
        paths=PathConfig(
            base_dir=base_path,
            data_dir=data_dir,
            artifact_dir=artifact_dir,
            cache_db_path=cache_db_path,
            index_dir=index_dir,
        ),
    )
