"""FAISS-backed vector storage and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import ChunkRecord, RetrievalResult
from .utils import ensure_directory


class FAISSVectorStore:
    """Stores chunk embeddings in a FAISS index with sidecar metadata."""

    def __init__(self) -> None:
        """Initializes an empty vector store."""

        self.index = None
        self.chunks: list[ChunkRecord] = []

    def build(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        """Builds a cosine-similarity FAISS index from chunk embeddings."""

        if len(chunks) != len(vectors):
            raise ValueError("Chunk count and vector count must match.")

        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise ImportError("FAISS vector storage requires the 'faiss-cpu' package.") from exc

        if not vectors:
            raise ValueError("Cannot build a vector store with zero vectors.")

        matrix = np.asarray(vectors, dtype="float32")
        dimension = matrix.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(matrix)
        self.chunks = chunks

    def save(self, directory: str | Path) -> None:
        """Persists the FAISS index and chunk metadata to disk."""

        if self.index is None:
            raise ValueError("Vector index has not been built yet.")

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Saving the FAISS index requires the 'faiss-cpu' package.") from exc

        directory_path = ensure_directory(directory)
        faiss.write_index(self.index, str(directory_path / "index.faiss"))
        metadata_path = directory_path / "chunks.json"
        metadata_path.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Loads a persisted FAISS index and chunk metadata from disk."""

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Loading the FAISS index requires the 'faiss-cpu' package.") from exc

        directory_path = Path(directory)
        store = cls()
        store.index = faiss.read_index(str(directory_path / "index.faiss"))

        chunk_payloads = json.loads((directory_path / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [ChunkRecord(**payload) for payload in chunk_payloads]
        return store

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Searches the index and filters results by minimum similarity score."""

        if self.index is None:
            raise ValueError("Vector index has not been built or loaded.")

        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("Searching the FAISS index requires 'numpy'.") from exc

        query_matrix = np.asarray([query_vector], dtype="float32")
        scores, indices = self.index.search(query_matrix, top_k)

        results: list[RetrievalResult] = []

        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            if float(score) < min_score:
                continue

            chunk = self.chunks[index]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=dict(chunk.metadata),
                )
            )

        return results
