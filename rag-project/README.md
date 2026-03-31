# Production RAG System

An end-to-end Retrieval-Augmented Generation system built to answer questions from documents and video transcripts using semantic search, FAISS, and configurable LLM backends.

## Project Summary

This project demonstrates a production-style RAG workflow that supports:

- document ingestion for TXT, CSV, and PDF files,
- transcript-based question answering for video content,
- local model execution,
- OpenAI-compatible API execution,
- Streamlit and notebook-based interaction.

For interviews, this project shows practical understanding of data pipelines, vector search, modular backend design, and configurable AI systems.

## Key Features

- Modular pipeline for loading, cleaning, chunking, embedding, indexing, retrieval, and generation
- FAISS-based vector search with metadata-aware retrieval
- Local video workflow using sample extraction, audio conversion, and Whisper transcription
- Dual execution modes: fully local or OpenAI-compatible API mode
- Streamlit UI, notebook workflow, and CLI entry points
- Source-aware outputs and environment-driven configuration

## Architecture

The core pipeline follows this sequence:

1. Data Loading
2. Data Cleaning
3. Chunking
4. Embedding
5. Vector Storage in FAISS
6. Semantic Retrieval
7. LLM Response Generation
8. Evaluation and inspection

Video processing extends the same pipeline by adding:

1. MP4 input
2. Sample extraction
3. WAV conversion
4. Whisper transcription
5. Transcript indexing
6. RAG-based question answering

## Repository Layout

```text
rag_system/
  config.py
  loaders.py
  preprocess.py
  chunking.py
  embeddings.py
  vector_store.py
  retriever.py
  generator.py
  transcription.py
  video_rag.py
app/
  api.py
  streamlit_app.py
data/
  sample/
  video_transcripts/
docs/
  PROJECT_DOCUMENTATION_HI_EN.md
notebooks/
  rag_step_by_step.py
run_local_video_rag.py
run_rag_cli.py
requirements.txt
```

## Running The Project

### Local mode

Use the local stack when you do not want API cost:

```powershell
python run_local_video_rag.py
```

Useful options:

```powershell
python run_local_video_rag.py --duration-seconds 60
python run_local_video_rag.py --duration-seconds 300 --query "What is the speaker teaching?"
python run_local_video_rag.py --start-seconds 60 --duration-seconds 120
```

### OpenAI-compatible API mode

1. Copy `.env.example` to `.env`
2. Fill `OPENAI_API_KEY`
3. Set `USE_OPENAI=true`
4. Run the same command:

```powershell
python run_local_video_rag.py
```

Recommended API-mode values:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
USE_OPENAI=true
VIDEO_EMBEDDING_PROVIDER=openai
VIDEO_EMBEDDING_MODEL=text-embedding-3-small
VIDEO_LLM_PROVIDER=openai
VIDEO_LLM_MODEL=gpt-4o-mini
```

## Streamlit UI

Launch the browser interface with:

```powershell
python -m streamlit run app/streamlit_app.py
```

The app supports:

- local or OpenAI-backed execution,
- video selection and sample duration control,
- question input,
- transcript preview,
- retrieved chunk inspection.

## Documentation

- Detailed bilingual documentation: `docs/PROJECT_DOCUMENTATION_HI_EN.md`
- Environment template: `.env.example`
- Notebook walkthrough: `notebooks/rag_step_by_step.py`

## Interview Value

This project is useful in interviews because it demonstrates:

- end-to-end RAG design rather than isolated scripts,
- modular Python engineering,
- local and hosted model interoperability,
- vector search and retrieval grounding,
- practical handling of real-world inputs such as video.
