# AI Conference Assistant (MVP)

MVP-система для записи встречи, распознавания речи, диаризации спикеров и генерации краткого резюме.

## Возможности

- Запись аудио с микрофона в WAV
- Подключение внешнего аудиофайла вместо записи
- Распознавание речи через `faster-whisper`
- Диаризация (разделение по спикерам) через `pyannote.audio`
- Сохранение результатов:
  - полный текст (`transcript_*.txt`)
  - разбивка по спикерам (`speakers_*.json`)
  - резюме (`summary_*.txt`)

## Структура

```text
assistant-konferents/
│
├── backend/
│   ├── main.py
│   ├── recorder.py
│   ├── transcriber.py
│   ├── diarizer.py
│   ├── summarizer.py
│   └── utils.py
│
├── outputs/
│   ├── audio/
│   ├── transcripts/
│   └── summaries/
│
├── requirements.txt
└── README.md
```

## Установка

1. Python 3.10+
2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Для diarization нужен Hugging Face токен с доступом к `pyannote/speaker-diarization-3.1`:

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
```

## Запуск

### Вариант 1: запись с микрофона

```bash
python backend/main.py --duration 60 --language ru
```

### Вариант 2: внешний аудиофайл

```bash
python backend/main.py --audio-file /path/to/meeting.wav --language ru
```

## Что сохраняется

В папках `outputs/transcripts` и `outputs/summaries`:

- `transcript_YYYYMMDD_HHMMSS.txt`
- `speakers_YYYYMMDD_HHMMSS.json`
- `summary_YYYYMMDD_HHMMSS.txt`

## Пример `speakers.json`

```json
[
  {"speaker": "A", "text": "Коллеги, начинаем встречу."},
  {"speaker": "B", "text": "По задаче интеграции есть обновление."},
  {"speaker": "A", "text": "Отлично, фиксируем дедлайн на пятницу."}
]
```

## Важные замечания

- Модели Whisper/Transformers/PyAnnote скачиваются при первом запуске.
- Для GPU можно адаптировать параметры в `transcriber.py` (`device`, `compute_type`).
- Качество summary зависит от выбранной модели и языка.
