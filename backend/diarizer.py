"""Speaker diarization and speaker-attributed transcript generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyannote.audio import Pipeline


class SpeakerDiarizer:
    """Performs diarization using pyannote pipeline from Hugging Face."""

    def __init__(self, hf_token: str, model_id: str = "pyannote/speaker-diarization-3.1") -> None:
        if not hf_token:
            raise ValueError("Hugging Face token is required for pyannote diarization")

        self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

    def diarize(self, audio_path: str | Path, transcript_segments: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Map transcript segments to speakers.

        Returns:
            A list like:
            [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}]
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        diarization = self.pipeline(str(audio_path))

        speaker_map: dict[str, str] = {}
        next_label_ord = ord("A")

        def map_speaker_label(raw_label: str) -> str:
            nonlocal next_label_ord
            if raw_label not in speaker_map:
                speaker_map[raw_label] = chr(next_label_ord)
                next_label_ord += 1
            return speaker_map[raw_label]

        result: list[dict[str, str]] = []

        for seg in transcript_segments:
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            seg_mid = (seg_start + seg_end) / 2

            chosen_raw_speaker = None
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= seg_mid <= turn.end:
                    chosen_raw_speaker = speaker
                    break

            if chosen_raw_speaker is None:
                chosen_speaker = "Unknown"
            else:
                chosen_speaker = map_speaker_label(str(chosen_raw_speaker))

            text = str(seg["text"]).strip()
            if text:
                result.append({"speaker": chosen_speaker, "text": text})

        # Merge neighboring entries by same speaker for readability.
        merged: list[dict[str, str]] = []
        for item in result:
            if merged and merged[-1]["speaker"] == item["speaker"]:
                merged[-1]["text"] += " " + item["text"]
            else:
                merged.append(item)

        return merged
