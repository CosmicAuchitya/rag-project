"""Shared data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DocumentRecord:
    """Represents a source document before chunking."""

    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    """Represents a document chunk stored in the vector index."""

    chunk_id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Represents a retrieved chunk and its similarity score."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerResult:
    """Represents the final generated answer plus provenance."""

    query: str
    answer: str
    retrieved_results: list[RetrievalResult]
    prompt: str


@dataclass(slots=True)
class EvaluationExample:
    """Defines one retrieval evaluation case."""

    query: str
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
