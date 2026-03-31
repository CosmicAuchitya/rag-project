"""Text cleaning and preprocessing utilities."""

from __future__ import annotations

import re

from .models import DocumentRecord
from .utils import normalize_whitespace


CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Cleans text by removing control characters and normalizing whitespace."""

    cleaned = CONTROL_CHAR_PATTERN.sub(" ", text)
    cleaned = cleaned.replace("\ufeff", " ")
    return normalize_whitespace(cleaned)


def preprocess_documents(documents: list[DocumentRecord]) -> list[DocumentRecord]:
    """Applies text cleaning to a collection of documents."""

    cleaned_documents: list[DocumentRecord] = []

    for document in documents:
        cleaned_text = clean_text(document.text)

        if cleaned_text:
            metadata = dict(document.metadata)
            metadata["cleaned"] = True
            cleaned_documents.append(DocumentRecord(text=cleaned_text, metadata=metadata))

    return cleaned_documents
