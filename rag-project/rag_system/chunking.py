"""Chunking utilities for splitting cleaned documents into retrievable units."""

from __future__ import annotations

import uuid

from .config import ChunkingConfig
from .models import ChunkRecord, DocumentRecord


def split_documents(documents: list[DocumentRecord], config: ChunkingConfig) -> list[ChunkRecord]:
    """Splits cleaned documents into overlapping chunks with source metadata."""

    chunks: list[ChunkRecord] = []

    for document in documents:
        text_fragments = _recursive_split(
            document.text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

        for index, fragment in enumerate(text_fragments):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    text=fragment,
                    metadata=metadata,
                )
            )

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...],
) -> list[str]:
    """Splits text recursively using progressively smaller separators."""

    if len(text) <= chunk_size:
        return [text]

    current_separator = separators[0]
    remaining_separators = separators[1:]

    if current_separator and current_separator in text:
        parts = text.split(current_separator)
        merged_parts = _merge_parts(parts, current_separator, chunk_size)
    elif remaining_separators:
        return _recursive_split(text, chunk_size, chunk_overlap, remaining_separators)
    else:
        merged_parts = [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    final_chunks: list[str] = []

    for part in merged_parts:
        if len(part) > chunk_size and remaining_separators:
            final_chunks.extend(_recursive_split(part, chunk_size, chunk_overlap, remaining_separators))
        else:
            final_chunks.append(part.strip())

    return _apply_overlap([chunk for chunk in final_chunks if chunk], chunk_overlap)


def _merge_parts(parts: list[str], separator: str, chunk_size: int) -> list[str]:
    """Merges split parts into bounded chunk candidates."""

    merged: list[str] = []
    buffer = ""

    for part in parts:
        candidate = f"{buffer}{separator}{part}" if buffer else part

        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = part

    if buffer:
        merged.append(buffer.strip())

    return merged


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Adds character-level overlap between adjacent chunks."""

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]

    for current_chunk in chunks[1:]:
        previous_tail = overlapped[-1][-overlap:]
        overlapped.append(f"{previous_tail} {current_chunk}".strip())

    return overlapped
