"""Embedding models and cache management."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from .config import EmbeddingConfig


class EmbeddingCache:
    """Caches embeddings in SQLite to avoid recomputing identical vectors."""

    def __init__(self, db_path: str | Path) -> None:
        """Initializes the cache database."""

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        """Creates the cache table if it does not already exist."""

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> list[float] | None:
        """Returns a cached embedding when available."""

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT vector_json FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        return json.loads(row[0]) if row else None

    def set(self, cache_key: str, vector: list[float]) -> None:
        """Stores an embedding vector in the cache."""

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO embedding_cache(cache_key, vector_json) VALUES(?, ?)",
                (cache_key, json.dumps(vector)),
            )


class BaseEmbedder(ABC):
    """Defines the shared interface for all embedders."""

    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache | None = None) -> None:
        """Stores configuration and cache references."""

        self.config = config
        self.cache = cache

    @abstractmethod
    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embeds raw texts without consulting the cache."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts while reusing cached vectors when possible."""

        results: list[list[float] | None] = [None] * len(texts)
        missing_pairs: list[tuple[int, str, str]] = []

        for index, text in enumerate(texts):
            cache_key = self._build_cache_key(text)

            if self.cache:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    results[index] = cached
                    continue

            missing_pairs.append((index, cache_key, text))

        if missing_pairs:
            fresh_vectors = self._embed_uncached([text for _, _, text in missing_pairs])

            for (index, cache_key, _), vector in zip(missing_pairs, fresh_vectors, strict=True):
                normalized_vector = _normalize_vector(vector) if self.config.normalize else vector
                results[index] = normalized_vector

                if self.cache:
                    self.cache.set(cache_key, normalized_vector)

        return [vector for vector in results if vector is not None]

    def _build_cache_key(self, text: str) -> str:
        """Builds a stable cache key from the provider, model, and text."""

        payload = f"{self.config.provider}|{self.config.model_name}|{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embeds text using a local Sentence Transformers model."""

    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache | None = None) -> None:
        """Loads the sentence transformer model lazily."""

        super().__init__(config=config, cache=cache)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Sentence Transformers support requires the 'sentence-transformers' package."
            ) from exc

        self.model = SentenceTransformer(config.model_name)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts using the configured sentence transformer model."""

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize,
        )
        return vectors.tolist()


class OpenAIEmbedder(BaseEmbedder):
    """Embeds text using OpenAI-compatible embedding APIs."""

    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache | None = None) -> None:
        """Initializes the OpenAI client."""

        super().__init__(config=config, cache=cache)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI embeddings require the 'openai' package.") from exc

        self.client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts using the configured OpenAI-compatible endpoint."""

        response = self.client.embeddings.create(model=self.config.model_name, input=texts)
        return [item.embedding for item in response.data]


def build_embedder(config: EmbeddingConfig, cache_db_path: str | Path) -> BaseEmbedder:
    """Builds the configured embedder and optional cache."""

    cache = EmbeddingCache(cache_db_path) if config.cache_enabled else None

    if config.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(config=config, cache=cache)
    if config.provider in {"openai", "openai_compatible"}:
        return OpenAIEmbedder(config=config, cache=cache)

    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def _normalize_vector(vector: list[float]) -> list[float]:
    """Normalizes a vector for cosine-similarity style retrieval."""

    squared_sum = sum(value * value for value in vector)
    magnitude = squared_sum ** 0.5

    if magnitude == 0:
        return vector

    return [value / magnitude for value in vector]
