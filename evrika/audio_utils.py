# evrika/audio_utils.py
"""
Audio utilities for Evrika Briefs:
- STT: transcribe_question_bytes -> short user question from microphone.
- TTS: synthesize_answer_tts -> short spoken answer as audio bytes.
"""

import io
from typing import Tuple

from pydub import AudioSegment

from .config import openai_client


def transcribe_question_bytes(
    audio_bytes: bytes,  
    model: str = "whisper-1",
) -> str:
    """
    Transcribe a short audio snippet (user question) using OpenAI Whisper.

    We accept whatever the browser records (e.g. m4a), decode it with pydub/ffmpeg,
    convert it to WAV in memory, and then send the WAV to Whisper.

    This avoids "Invalid file format" errors for tricky m4a encodings.
    """
    print(f"[VOICE] transcribe_question_bytes: received {len(audio_bytes)} bytes")

    # 1) Decode the incoming audio (m4a, webm, etc.) with pydub
    try:
        input_buffer = io.BytesIO(audio_bytes)
        input_buffer.seek(0)
        # Let pydub/ffmpeg auto-detect the format
        audio = AudioSegment.from_file(input_buffer)
    except Exception as e:
        print(f"[VOICE][STT DECODE ERROR] {e}")
        # Bubble up so /voice-query returns a clear STT error JSON
        raise

    # 2) Re-encode as WAV in memory
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    # Give the buffer a filename with extension so Whisper is happy
    wav_buffer.name = "voice.wav"

    # 3) Send WAV to Whisper
    transcript = openai_client.audio.transcriptions.create(
        model=model,
        file=wav_buffer,
    )
    text = getattr(transcript, "text", "") or ""
    return text.strip()


def synthesize_answer_tts(
    text: str,
    model: str = "gpt-4o-mini-tts",
    voice: str = "nova",
) -> Tuple[bytes, str]:
    """
    Turn a short text answer into speech audio bytes using OpenAI TTS.

    Returns:
        (audio_bytes, mime_type)
    """
    if not text.strip():
        # Return empty audio if there's nothing to say
        return b"", "audio/mpeg"

    # Call OpenAI TTS
    speech = openai_client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )

    # openai>=1.6 style: speech is a streaming-like object with .read()
    # If your version differs, you can adapt this to speech.content instead.
    audio_bytes = speech.read()

    return audio_bytes, "audio/mpeg"
