"""Entry point for AI Conference Assistant MVP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from diarizer import SpeakerDiarizer
from recorder import record_audio
from summarizer import MeetingSummarizer
from transcriber import WhisperTranscriber
from utils import ensure_dirs, save_json, save_text, timestamp_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Conference Assistant")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to existing audio file (WAV/MP3).")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds if no external file.")
    parser.add_argument("--language", type=str, default="ru", help="Language code for Whisper (e.g., ru, en).")
    parser.add_argument("--whisper-model", type=str, default="small", help="faster-whisper model size.")
    parser.add_argument("--summary-model", type=str, default="google/mt5-small", help="Transformers model for summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    out_audio = base_dir / "outputs" / "audio"
    out_trans = base_dir / "outputs" / "transcripts"
    out_summ = base_dir / "outputs" / "summaries"
    ensure_dirs(out_audio, out_trans, out_summ)

    run_id = timestamp_now()
    audio_path = out_audio / f"meeting_{run_id}.wav"

    try:
        if args.audio_file:
            source_audio = Path(args.audio_file)
            if not source_audio.exists():
                raise FileNotFoundError(f"External audio file not found: {source_audio}")
            audio_path = source_audio
            print(f"[Main] Using external audio file: {audio_path}")
        else:
            audio_path = record_audio(audio_path, duration_sec=args.duration)
            print(f"[Main] Audio saved: {audio_path}")

        transcriber = WhisperTranscriber(model_size=args.whisper_model)
        transcription = transcriber.transcribe(audio_path=audio_path, language=args.language)
        full_text = transcription["text"]
        print("[Main] Transcription done.")

        hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        diarizer = SpeakerDiarizer(hf_token=hf_token)
        speaker_segments = diarizer.diarize(audio_path=audio_path, transcript_segments=transcription["segments"])
        print("[Main] Diarization done.")

        summarizer = MeetingSummarizer(model_name=args.summary_model)
        summary_text = summarizer.summarize(full_text)
        print("[Main] Summary done.")

        transcript_path = out_trans / f"transcript_{run_id}.txt"
        speakers_path = out_trans / f"speakers_{run_id}.json"
        summary_path = out_summ / f"summary_{run_id}.txt"

        save_text(transcript_path, full_text)
        save_json(speakers_path, speaker_segments)
        save_text(summary_path, summary_text)

        print("[Main] Files saved:")
        print(f" - Transcript: {transcript_path}")
        print(f" - Speakers JSON: {speakers_path}")
        print(f" - Summary: {summary_path}")

    except Exception as exc:
        print(f"[Error] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
