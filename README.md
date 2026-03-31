# Production RAG System

This repository contains a complete Retrieval-Augmented Generation project for question answering over documents and video transcripts.

The full implementation lives inside the [`rag-project`](./rag-project) folder and includes:

- document ingestion for TXT, CSV, and PDF files
- FAISS-based semantic retrieval
- local and OpenAI-compatible generation workflows
- video transcription and transcript-based question answering
- Streamlit and API entry points

## Project Highlights

- Modular Python pipeline for loading, cleaning, chunking, embedding, retrieval, and generation
- Local-first workflow with optional API-backed execution
- Interview-ready structure with reusable components and documentation
- Support for notebooks, CLI runs, and browser-based demos

## Quick Start

1. Open the main project folder: [`rag-project`](./rag-project)
2. Install dependencies from `requirements.txt`
3. Review the setup instructions in [`rag-project/README.md`](./rag-project/README.md)
4. Run the local workflow or Streamlit app

## Main Files

- [`rag-project/README.md`](./rag-project/README.md)
- [`rag-project/run_local_video_rag.py`](./rag-project/run_local_video_rag.py)
- [`rag-project/run_rag_cli.py`](./rag-project/run_rag_cli.py)
- [`rag-project/app/streamlit_app.py`](./rag-project/app/streamlit_app.py)
- [`rag-project/docs/PROJECT_DOCUMENTATION_HI_EN.md`](./rag-project/docs/PROJECT_DOCUMENTATION_HI_EN.md)
