# Local Video RAG - Detailed Project Documentation

## 1. Executive Summary

This project is a modular Retrieval-Augmented Generation system designed to answer questions from both traditional documents and video transcripts. It combines preprocessing, embeddings, FAISS-based retrieval, and configurable answer generation into one reusable pipeline.

Hindi note:
Ye project ek structured RAG system hai jo documents aur video transcript dono par question answering kar sakta hai.

## 2. Problem Statement

The original goal was not only to transcribe a video, but to build a system that could:

- understand source material,
- retrieve the most relevant information,
- answer user questions from grounded context,
- support both local execution and API-backed execution.

This matters because raw transcripts or large documents are difficult to use directly in a reliable question-answering flow.

Hindi note:
Sirf transcript banana enough nahi tha. Humein aisa system chahiye tha jo relevant information dhoondh kar context-based answer de.

## 3. Why RAG Was The Right Approach

RAG was chosen because it solves an important limitation of direct prompting: long documents and long transcripts do not fit cleanly into a single prompt, and even when they do, answer quality can become inconsistent.

The RAG pipeline improves this by:

- splitting content into meaningful chunks,
- embedding those chunks into vector space,
- retrieving only the most relevant chunks for a question,
- generating an answer from retrieved evidence.

Hindi note:
RAG ne is problem ko solve kiya kyunki poora transcript model ko dene ke bajay sirf relevant chunks use kiye gaye.

## 4. System Design

The final system follows this flow:

1. Load files or transcripts
2. Clean the text
3. Split content into chunks
4. Generate embeddings
5. Store vectors in FAISS
6. Retrieve top-k relevant chunks
7. Generate a grounded answer
8. Inspect or evaluate results

For videos, the workflow adds:

1. MP4 sample extraction
2. Audio conversion
3. Whisper transcription
4. Transcript persistence as TXT and JSON

## 5. Key Engineering Choices

### 5.1 Modular file structure

The project was intentionally split into focused modules such as `loaders.py`, `embeddings.py`, `vector_store.py`, `generator.py`, and `transcription.py`.

Why this was useful:

- each file has one clear responsibility,
- debugging becomes easier,
- local and API-backed modes can coexist cleanly,
- future changes stay isolated instead of breaking the full pipeline.

Hindi note:
Ye structure interview aur maintenance dono ke liye strong hai, kyunki har part alag file me controlled way me rakha gaya hai.

### 5.2 FAISS for retrieval

FAISS was chosen because it is lightweight, fast, and practical for local semantic search.

### 5.3 Sentence Transformers for local embeddings

`sentence-transformers/all-MiniLM-L6-v2` was selected because it is fast and efficient while still being strong enough for semantic retrieval.

### 5.4 Whisper for local transcription

`openai/whisper-tiny.en` was used for the first working version because it keeps runtime manageable for CPU-based execution.

Tradeoff:
larger Whisper models may produce better transcription quality, but require more time and resources.

### 5.5 FLAN-T5 small for local answering

`google/flan-t5-small` was chosen as a practical local answer model to keep the project runnable without requiring a large GPU setup.

Tradeoff:
the model is usable, but not as strong as larger hosted or quantized instruct models.

### 5.6 OpenAI-compatible mode

The project was extended to support OpenAI-compatible API usage so that the same codebase can be demonstrated in both:

- no-cost local mode,
- higher-quality API-backed mode.

This is important for interviews because it shows engineering flexibility rather than hard-coding a single path.

## 6. Challenges Faced

### Challenge 1: No paid API requirement

The initial constraint was to avoid paid APIs, which meant the entire workflow had to run locally.

### Challenge 2: No global ffmpeg

The machine did not have a globally installed ffmpeg binary. This was handled with `imageio-ffmpeg`, which provides a bundled executable.

### Challenge 3: Video duration

The source video was long enough that full processing on CPU could be slow. A 60-second sample workflow was created first to validate the pipeline safely.

### Challenge 4: Model-size tradeoffs

Local models are often limited by CPU speed, RAM, and disk usage. Smaller models were selected first to ensure the project could actually run.

### Challenge 5: Runtime compatibility

Some library and provider combinations required extra handling, especially around local generation paths and environment-based configuration.

## 7. What Was Implemented

The project now includes:

- document RAG for TXT, CSV, and PDF,
- video-to-transcript processing,
- local semantic retrieval,
- local answer generation,
- OpenAI-compatible answer generation,
- CLI usage,
- notebook usage,
- Streamlit interface,
- API-ready backend layer.

## 8. Output Artifacts

The workflow generates and stores:

- sample video clips,
- extracted WAV files,
- transcript text files,
- transcript metadata JSON,
- FAISS indexes,
- chunk metadata,
- embedding cache files.

Primary directories:

- `artifacts/video_processing`
- `artifacts/faiss_index`
- `data/video_transcripts`

## 9. Streamlit Support

Yes, the system supports Streamlit through `app/streamlit_app.py`.

What Streamlit provides:

- simple browser UI,
- easy demo flow,
- support for local mode and API mode,
- transcript preview,
- retrieved-chunk visibility.

Command:

```powershell
python -m streamlit run app/streamlit_app.py
```

Hindi note:
Streamlit interviewer demo ke liye useful hai, kyunki bina code khole system ka end-to-end flow dikhaya ja sakta hai.

## 10. OpenAI API Demo Support

To make the project interview-friendly, an explicit API-key path was added.

Where the key goes:

- copy `.env.example` to `.env`
- fill `OPENAI_API_KEY`
- set `USE_OPENAI=true`

Recommended settings:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
USE_OPENAI=true
VIDEO_EMBEDDING_PROVIDER=openai
VIDEO_EMBEDDING_MODEL=text-embedding-3-small
VIDEO_LLM_PROVIDER=openai
VIDEO_LLM_MODEL=gpt-4o-mini
```

Files that read or use these settings:

- `run_local_video_rag.py`
- `notebooks/rag_step_by_step.py`
- `app/api.py`
- `app/streamlit_app.py`
- `rag_system/video_rag.py`

## 11. Recruiter / Interview Summary

This project demonstrates:

- applied understanding of RAG system design,
- practical experience with vector databases,
- configurable AI architecture,
- local and hosted model interoperability,
- real-world multimodal preprocessing,
- interface support through notebooks, CLI, and Streamlit.

Short version for interviews:

"Built a modular end-to-end RAG system for documents and video transcripts using FAISS, configurable embeddings, and both local and OpenAI-compatible generation paths, with Streamlit support for interactive demos."

## 12. Current Limitations

- The local answer model is intentionally small, so responses can be brief.
- Whisper tiny is optimized for speed rather than maximum transcription quality.
- Full-length video processing is slower on CPU.
- The local-first setup is practical, but stronger models can improve answer quality.

## 13. Recommended Improvements

- Upgrade transcription to `whisper-base` or `whisper-small`
- Add timestamp-aware chunk metadata
- Add conversation memory
- Improve transcript exploration in the UI
- Add stronger local inference options such as quantized instruct models

## 14. Final Takeaway

This is not just a script collection. It is a structured RAG project that shows clear engineering thinking, modular implementation, deployment flexibility, and interview-ready presentation.

Hindi note:
Is project ka strongest point ye hai ki ye sirf kaam nahi karta, balki achhe engineering structure ke saath present bhi hota hai.
