"""Local video and audio transcription helpers."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from .utils import ensure_directory


def resolve_ffmpeg_executable() -> str:
    """Returns the bundled ffmpeg executable path provided by imageio-ffmpeg."""

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ImportError(
            "Video processing requires the 'imageio-ffmpeg' package."
        ) from exc

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_video_sample(
    video_path: str | Path,
    output_dir: str | Path,
    start_seconds: int = 0,
    duration_seconds: int = 60,
) -> tuple[Path, Path]:
    """Creates a short sample video and a mono 16k WAV file from the source video."""

    video_file = Path(video_path)
    target_dir = ensure_directory(output_dir)
    ffmpeg_exe = resolve_ffmpeg_executable()

    sample_stem = f"{video_file.stem}_sample_{duration_seconds}s"
    sample_video_path = target_dir / f"{sample_stem}.mp4"
    sample_audio_path = target_dir / f"{sample_stem}.wav"

    video_command = [
        ffmpeg_exe,
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(video_file),
        "-t",
        str(duration_seconds),
        "-c",
        "copy",
        str(sample_video_path),
    ]
    audio_command = [
        ffmpeg_exe,
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(video_file),
        "-t",
        str(duration_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(sample_audio_path),
    ]

    _run_ffmpeg(video_command)
    _run_ffmpeg(audio_command)
    return sample_video_path, sample_audio_path


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "openai/whisper-tiny.en",
    chunk_length_s: int = 30,
) -> dict[str, object]:
    """Transcribes a WAV file locally with a Hugging Face Whisper pipeline."""

    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Local transcription requires the 'torch' and 'transformers' packages."
        ) from exc

    audio_array, sampling_rate = _read_wav_file(audio_path)
    device = 0 if torch.cuda.is_available() else -1

    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    result = asr_pipeline(
        {"array": audio_array, "sampling_rate": sampling_rate},
        chunk_length_s=chunk_length_s,
        return_timestamps=True,
    )
    return {
        "text": str(result.get("text", "")).strip(),
        "chunks": result.get("chunks", []),
        "model_name": model_name,
        "sampling_rate": sampling_rate,
    }


def save_transcript_artifacts(
    transcription: dict[str, object],
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Saves transcript text and segment metadata for downstream RAG ingestion."""

    target_dir = ensure_directory(output_dir)
    transcript_path = target_dir / f"{stem}.txt"
    metadata_path = target_dir / f"{stem}.json"

    transcript_path.write_text(str(transcription["text"]).strip(), encoding="utf-8")
    metadata_path.write_text(json.dumps(transcription, indent=2, ensure_ascii=True), encoding="utf-8")
    return transcript_path, metadata_path


def _read_wav_file(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Reads PCM WAV audio into a float32 numpy array."""

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        channel_count = wav_file.getnchannels()
        raw_frames = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported.")

    audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)

    return audio, sample_rate


def _run_ffmpeg(command: list[str]) -> None:
    """Runs an ffmpeg command and surfaces stderr on failure."""

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {result.stderr}")
