"""Speech-to-text module powered by faster-whisper."""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class WhisperTranscriber:
    """Wrapper around faster-whisper model."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str | Path, language: str = "ru") -> dict:
        """Transcribe a WAV/MP3 file into text plus timestamped segments."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(str(audio_path), language=language, vad_filter=True)

        seg_list: list[dict] = []
        full_text_parts: list[str] = []

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            full_text_parts.append(text)
            seg_list.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )

        return {
            "language": info.language,
            "duration": float(info.duration),
            "text": " ".join(full_text_parts).strip(),
            "segments": seg_list,
        }
