"""End-to-end orchestration for the RAG system."""

from __future__ import annotations

from pathlib import Path

from .chunking import split_documents
from .config import PipelineConfig
from .embeddings import build_embedder
from .evaluator import evaluate_retrieval
from .generator import AnswerGenerator
from .loaders import load_documents
from .models import AnswerResult, EvaluationExample
from .preprocess import preprocess_documents
from .retriever import SemanticRetriever
from .utils import debug_print, ensure_directory, sanitize_collection_name, set_reproducibility
from .vector_store import FAISSVectorStore


class RAGPipeline:
    """Coordinates data ingestion, indexing, retrieval, and answer generation."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initializes the pipeline with deterministic settings."""

        self.config = config
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.generator = AnswerGenerator(config.llm)
        ensure_directory(config.paths.artifact_dir)
        set_reproducibility(config.random_seed)

    def build_index(self, input_paths: list[str | Path]) -> dict[str, int]:
        """Builds and persists a FAISS index from input documents."""

        debug_print(self.config.debug, "Loading documents.")
        documents = load_documents([Path(path) for path in input_paths])

        debug_print(self.config.debug, f"Loaded {len(documents)} raw documents.")
        cleaned_documents = preprocess_documents(documents)
        debug_print(self.config.debug, f"Retained {len(cleaned_documents)} cleaned documents.")

        chunks = split_documents(cleaned_documents, self.config.chunking)
        debug_print(self.config.debug, f"Generated {len(chunks)} chunks.")

        self.embedder = build_embedder(
            config=self.config.embeddings,
            cache_db_path=self.config.paths.cache_db_path,
        )
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        debug_print(self.config.debug, f"Embedded {len(vectors)} chunks.")

        self.vector_store = FAISSVectorStore()
        self.vector_store.build(chunks=chunks, vectors=vectors)

        index_dir = self._collection_index_dir()
        self.vector_store.save(index_dir)
        debug_print(self.config.debug, f"Saved FAISS index to {index_dir}.")

        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            config=self.config.retrieval,
        )

        return {
            "raw_documents": len(documents),
            "cleaned_documents": len(cleaned_documents),
            "chunks": len(chunks),
            "vectors": len(vectors),
        }

    def load_index(self) -> None:
        """Loads a persisted FAISS index and reconstructs retriever dependencies."""

        self.embedder = build_embedder(
            config=self.config.embeddings,
            cache_db_path=self.config.paths.cache_db_path,
        )
        self.vector_store = FAISSVectorStore.load(self._collection_index_dir())
        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            config=self.config.retrieval,
        )
        debug_print(self.config.debug, "Loaded persisted FAISS index.")

    def answer_query(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AnswerResult:
        """Retrieves relevant context and generates a grounded answer."""

        if self.retriever is None:
            raise ValueError("Retriever is not ready. Build or load the index first.")

        debug_print(self.config.debug, f"Running retrieval for query: {query}")
        results = self.retriever.retrieve(query=query, top_k=top_k, min_score=min_score)
        debug_print(self.config.debug, f"Retrieved {len(results)} chunks.")

        answer = self.generator.generate_answer(query=query, retrieved_results=results)
        debug_print(self.config.debug, "Generated final answer.")
        return answer

    def evaluate(self, examples: list[EvaluationExample], top_k: int | None = None) -> dict[str, object]:
        """Evaluates retrieval quality on labeled examples."""

        if self.retriever is None:
            raise ValueError("Retriever is not ready. Build or load the index first.")

        return evaluate_retrieval(
            retriever=self.retriever,
            examples=examples,
            top_k=top_k or self.config.retrieval.top_k,
        )

    def _collection_index_dir(self) -> Path:
        """Builds the storage path for the configured collection."""

        collection_name = sanitize_collection_name(self.config.collection_name)
        return self.config.paths.index_dir / collection_name
