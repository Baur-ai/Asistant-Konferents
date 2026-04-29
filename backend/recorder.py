"""Audio recording utilities for AI Conference Assistant."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


def record_audio(
    output_path: str | Path,
    duration_sec: int = 60,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """Record audio from microphone and save it as WAV.

    Args:
        output_path: WAV file path to write.
        duration_sec: Recording duration in seconds.
        sample_rate: Audio sample rate.
        channels: Number of audio channels.

    Returns:
        Path to saved WAV file.

    Raises:
        ValueError: If duration or audio params are invalid.
        RuntimeError: If recording or saving fails.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[Recorder] Recording {duration_sec}s... Speak now.")
        audio = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()
        print("[Recorder] Recording finished.")
    except Exception as exc:
        raise RuntimeError(f"Failed to record audio: {exc}") from exc

    # Convert float [-1,1] to int16 PCM.
    audio_pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    try:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # int16 = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_pcm.tobytes())
    except Exception as exc:
        raise RuntimeError(f"Failed to save WAV to {output_path}: {exc}") from exc

    if not output_path.exists() or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"WAV file was not created correctly: {output_path}")

    return output_path
