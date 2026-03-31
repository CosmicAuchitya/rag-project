# Production RAG System

This project provides a modular Retrieval-Augmented Generation (RAG) pipeline designed for notebook workflows and production-style reuse.

## Resume / Interview Summary

This project demonstrates an end-to-end RAG system that supports document ingestion, video transcription, semantic retrieval with FAISS, and grounded answer generation using both local models and OpenAI-compatible APIs. It is designed as a modular production-style pipeline with configurable embeddings, configurable LLM providers, source-aware retrieval, and optional Streamlit and API interfaces.

## Architecture

The system follows this pipeline:

1. Data loading
2. Data cleaning
3. Chunking
4. Embedding
5. FAISS vector storage
6. Retrieval
7. LLM response generation
8. Retrieval evaluation

## Project Structure

```text
rag_system/
  config.py
  models.py
  loaders.py
  preprocess.py
  chunking.py
  embeddings.py
  vector_store.py
  retriever.py
  generator.py
  evaluator.py
  pipeline.py
app/
  api.py
data/
  sample/
notebooks/
  rag_step_by_step.py
requirements.txt
```

## Notes

- Python `3.11` or `3.12` is recommended for the smoothest FAISS compatibility.
- The code uses lazy imports for optional dependencies such as `faiss`, `openai`, `pypdf`, and `sentence-transformers`.
- The notebook-style script uses `# %%` cells so it can be executed step-by-step in VS Code or converted easily into a Jupyter notebook.
- In restricted environments, you can install packages into a repo-local `vendor_pkgs` folder and the notebook, CLI, and API entry points will pick it up automatically.

## Quick Start

1. Create and activate a Python environment.
2. Install the dependencies in `requirements.txt`.
3. If you want API mode, copy `.env.example` to `.env` and paste your API key in `OPENAI_API_KEY`.
4. Run the notebook-style script in `notebooks/rag_step_by_step.py`.

## Documentation

- Detailed Hindi-English project documentation: `docs/PROJECT_DOCUMENTATION_HI_EN.md`
- Interview/demo env template: `.env.example`

## Local Video Workflow

If you do not want to use any API, run the fully local video pipeline:

```powershell
python run_local_video_rag.py
```

That workflow:

1. extracts a sample clip from your MP4,
2. transcribes it locally with Whisper,
3. saves the transcript to `data/video_transcripts`,
4. builds a FAISS index over the transcript,
5. answers a question with local embeddings plus a local Hugging Face model.

Useful options:

```powershell
python run_local_video_rag.py --duration-seconds 60
python run_local_video_rag.py --duration-seconds 300 --query "What is the speaker teaching?"
python run_local_video_rag.py --start-seconds 60 --duration-seconds 120
```

If you want to show API mode in an interview:

1. Copy `.env.example` to `.env`
2. Put your key in `OPENAI_API_KEY=...`
3. Set `USE_OPENAI=true`
4. Run `python run_local_video_rag.py`

Important interview point:

- `.env.example` shows exactly where the key goes
- `run_local_video_rag.py` auto-loads `.env`
- `rag_system/video_rag.py` switches to OpenAI when `USE_OPENAI=true`
- `notebooks/rag_step_by_step.py` also reads the same environment variables

## Streamlit UI

Yes, this system can run on Streamlit.

Run:

```powershell
python -m streamlit run app/streamlit_app.py
```

The Streamlit app lets you:

1. choose the local video,
2. set sample duration,
3. ask a question,
4. view the transcript,
5. inspect retrieved chunks and the final answer.

## Environment Variables

- `EMBEDDING_PROVIDER`: `sentence_transformers`, `openai`, or `openai_compatible`
- `EMBEDDING_MODEL`: embedding model name
- `LLM_PROVIDER`: `extractive`, `openai`, or `openai_compatible`
- `LLM_MODEL`: generation model name
- `OPENAI_API_KEY`: API key for OpenAI-compatible providers
- `OPENAI_BASE_URL`: base URL for a local OpenAI-compatible server such as LM Studio or Ollama gateways
- `USE_OPENAI`: `true` for OpenAI mode in the video workflow
- `VIDEO_EMBEDDING_PROVIDER`: `sentence_transformers` or `openai`
- `VIDEO_EMBEDDING_MODEL`: video-RAG embedding model
- `VIDEO_LLM_PROVIDER`: `huggingface_local` or `openai`
- `VIDEO_LLM_MODEL`: video-RAG answer model
