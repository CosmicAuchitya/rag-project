"""Streamlit UI for the local video-to-RAG workflow."""

from __future__ import annotations

from pathlib import Path
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

import streamlit as st

from rag_system.video_rag import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LOCAL_LLM_MODEL_NAME,
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    DEFAULT_OPENAI_LLM_MODEL_NAME,
    DEFAULT_QUERY,
    DEFAULT_SAMPLE_DURATION_SECONDS,
    DEFAULT_VIDEO_FILE_NAME,
    DEFAULT_WHISPER_MODEL_NAME,
    run_local_video_rag,
)


st.set_page_config(page_title="Local Video RAG", layout="wide")
st.title("Local Video RAG")
st.caption("Supports both local mode and OpenAI API mode.")


def _as_bool(value: str) -> bool:
    """Converts a string flag into a boolean."""

    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    """Renders the Streamlit app."""

    with st.sidebar:
        st.header("Settings")
        video_name = st.text_input("Video file name", value=DEFAULT_VIDEO_FILE_NAME)
        start_seconds = st.number_input("Start seconds", min_value=0, value=0, step=1)
        duration_seconds = st.number_input(
            "Duration seconds",
            min_value=30,
            max_value=3600,
            value=DEFAULT_SAMPLE_DURATION_SECONDS,
            step=30,
        )
        whisper_model = st.text_input("Whisper model", value=DEFAULT_WHISPER_MODEL_NAME)
        use_openai = st.checkbox("Use OpenAI API", value=_as_bool(os.getenv("USE_OPENAI", "false")))
        default_embedding_model = (
            os.getenv("VIDEO_EMBEDDING_MODEL")
            or (DEFAULT_OPENAI_EMBEDDING_MODEL_NAME if use_openai else DEFAULT_EMBEDDING_MODEL_NAME)
        )
        default_llm_model = (
            os.getenv("VIDEO_LLM_MODEL")
            or (DEFAULT_OPENAI_LLM_MODEL_NAME if use_openai else DEFAULT_LOCAL_LLM_MODEL_NAME)
        )
        embedding_model = st.text_input("Embedding model", value=default_embedding_model)
        local_llm_model = st.text_input("Answer model", value=default_llm_model)
        query = st.text_area("Question", value=DEFAULT_QUERY, height=120)
        run_button = st.button("Process Video And Ask", type="primary")

    st.markdown(
        """
        **Workflow**

        1. Extract a sample from the local video
        2. Convert it to audio
        3. Transcribe locally with Whisper
        4. Build a FAISS index over the transcript
        5. Answer your question with local RAG
        """
    )

    if run_button:
        try:
            with st.spinner("Processing video, transcribing, indexing, and answering..."):
                result = run_local_video_rag(
                    project_dir=PROJECT_DIR,
                    video_name=video_name,
                    start_seconds=int(start_seconds),
                    duration_seconds=int(duration_seconds),
                    whisper_model=whisper_model,
                    embedding_provider="openai" if use_openai else "sentence_transformers",
                    embedding_model=embedding_model,
                    llm_provider="openai" if use_openai else "huggingface_local",
                    local_llm_model=local_llm_model,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_base_url=os.getenv("OPENAI_BASE_URL"),
                    query=query,
                    debug=False,
                )
        except Exception as exc:
            st.error(f"Failed to run local video RAG: {exc}")
            return

        st.success("Processing complete.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Artifacts")
            st.write(f"Sample video: `{result.sample_video_path}`")
            st.write(f"Sample audio: `{result.sample_audio_path}`")
            st.write(f"Transcript: `{result.transcript_path}`")
            st.write(f"Metadata: `{result.transcript_metadata_path}`")
        with col2:
            st.subheader("Index Stats")
            st.json(result.index_stats)

        st.subheader("Answer")
        st.write(result.answer_result.answer)

        st.subheader("Transcript Preview")
        st.text_area(
            "Transcript",
            value=result.transcription_text[:4000],
            height=240,
        )

        st.subheader("Retrieved Chunks")
        for item in result.answer_result.retrieved_results:
            st.markdown(
                f"**Score:** {item.score:.4f}  \n"
                f"**Source:** `{item.metadata.get('source')}`  \n"
                f"**Chunk Index:** `{item.metadata.get('chunk_index')}`"
            )
            st.write(item.text)
            st.divider()


if __name__ == "__main__":
    main()
