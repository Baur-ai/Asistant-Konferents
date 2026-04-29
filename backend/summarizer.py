"""Meeting summarization module using transformers summarization pipeline."""

from __future__ import annotations

from transformers import pipeline


class MeetingSummarizer:
    """Creates structured short summary: ideas, tasks, decisions."""

    def __init__(self, model_name: str = "google/mt5-small") -> None:
        # text2text-generation works for instruction-style summarization.
        self.generator = pipeline("text2text-generation", model=model_name)

    def summarize(self, transcript_text: str, max_new_tokens: int = 220) -> str:
        if not transcript_text.strip():
            raise ValueError("Cannot summarize empty transcript")

        prompt = (
            "Сделай краткое резюме встречи на русском языке в формате:\n"
            "1) Основные идеи\n2) Задачи\n3) Решения\n\n"
            "Текст встречи:\n"
            f"{transcript_text[:8000]}"
        )

        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
        return result[0]["generated_text"].strip()
