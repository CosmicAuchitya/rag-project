"""Semantic retrieval helpers built on top of the vector store."""

from __future__ import annotations

from .config import RetrievalConfig
from .embeddings import BaseEmbedder
from .models import RetrievalResult
from .preprocess import clean_text
from .vector_store import FAISSVectorStore


class SemanticRetriever:
    """Embeds a user query and retrieves the most relevant chunks."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: FAISSVectorStore,
        config: RetrievalConfig,
    ) -> None:
        """Stores retriever dependencies."""

        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieves the highest scoring chunks for the supplied query."""

        cleaned_query = clean_text(query)
        query_vector = self.embedder.embed_texts([cleaned_query])[0]
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k or self.config.top_k,
            min_score=self.config.min_score if min_score is None else min_score,
        )
