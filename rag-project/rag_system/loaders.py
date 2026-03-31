"""Document loading utilities for supported file formats."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import DocumentRecord


SUPPORTED_EXTENSIONS = {".txt", ".csv", ".pdf"}


def load_documents(paths: list[str | Path]) -> list[DocumentRecord]:
    """Loads TXT, CSV, and PDF documents from the provided paths."""

    documents: list[DocumentRecord] = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    documents.extend(load_documents([file_path]))
            continue

        suffix = path.suffix.lower()

        if suffix == ".txt":
            documents.extend(_load_txt(path))
        elif suffix == ".csv":
            documents.extend(_load_csv(path))
        elif suffix == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            raise ValueError(f"Unsupported file type: {path}")

    return documents


def _load_txt(path: Path) -> list[DocumentRecord]:
    """Loads a plain text file as a single document."""

    text = path.read_text(encoding="utf-8")
    return [
        DocumentRecord(
            text=text,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": "txt",
            },
        )
    ]


def _load_csv(path: Path) -> list[DocumentRecord]:
    """Loads each CSV row as an individual document for granular retrieval."""

    documents: list[DocumentRecord] = []

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)

        for row_number, row in enumerate(reader, start=1):
            row_text = " | ".join(f"{key}: {value}" for key, value in row.items())
            documents.append(
                DocumentRecord(
                    text=row_text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "file_type": "csv",
                        "row_number": row_number,
                    },
                )
            )

    return documents


def _load_pdf(path: Path) -> list[DocumentRecord]:
    """Loads a PDF file one page at a time to preserve metadata and locality."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires the 'pypdf' package. Install it from requirements.txt."
        ) from exc

    reader = PdfReader(str(path))
    documents: list[DocumentRecord] = []

    for page_number, page in enumerate(reader.pages, start=1):
        documents.append(
            DocumentRecord(
                text=page.extract_text() or "",
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": "pdf",
                    "page_number": page_number,
                },
            )
        )

    return documents
