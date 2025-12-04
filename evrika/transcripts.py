# evrika/transcripts.py
"""
YouTube metadata, audio download, and transcript helpers.
"""

import os
from typing import List, Tuple, Optional

from yt_dlp import YoutubeDL
from pydub import AudioSegment

from .config import openai_client

# Try to import YouTubeTranscriptApi safely
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
    HAS_YT_TRANSCRIPT_API = hasattr(YouTubeTranscriptApi, "get_transcript")
    if not HAS_YT_TRANSCRIPT_API:
        print(
            "[TRANSCRIPT] youtube-transcript-api imported, "
            "but YouTubeTranscriptApi has no get_transcript(). "
            "Likely wrong version or local module shadowing the package."
        )
except Exception as e:  # pragma: no cover
    print(f"[TRANSCRIPT] Could not import youtube_transcript_api: {e}")
    HAS_YT_TRANSCRIPT_API = False


# ---------------------------------------------------------------------------
# YT-DLP METADATA + AUDIO
# ---------------------------------------------------------------------------


def fetch_metadata_with_ytdlp(video_url: str) -> dict:
    """Fetch video metadata (ID, title, etc.) without downloading media."""
    ydl_opts = {"quiet": True, "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info


def fetch_audio_with_ytdlp(video_url: str) -> Tuple[str, dict]:
    """
    Download best audio from a YouTube URL as .mp3 and return:
    (audio_file_path, metadata_dict)

    We don't need studio-quality audio for transcription, so we keep the
    bitrate modest to reduce file size and speed up uploads.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "outtmpl": "%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",  # lower than 192 to speed things up
            }
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_path = ydl.prepare_filename(info)

    # Ensure .mp3 extension
    if not audio_path.endswith(".mp3"):
        audio_path = audio_path.rsplit(".", 1)[0] + ".mp3"

    return audio_path, info


# ---------------------------------------------------------------------------
# TRANSCRIPT HELPERS
# ---------------------------------------------------------------------------


def try_fetch_transcript_via_api(youtube_id: str) -> Optional[str]:
    """
    Try to fetch transcript via YouTubeTranscriptApi.
    Returns the full transcript text, or None if not available or library is broken.
    """
    if not HAS_YT_TRANSCRIPT_API:
        print("[TRANSCRIPT] YouTubeTranscriptApi not available or missing get_transcript() – skipping CC API.")
        return None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            youtube_id,
            languages=["en", "en-US", "en-GB"],
        )
        text = " ".join(
            entry["text"].replace("\n", " ").strip()
            for entry in transcript_list
            if entry["text"].strip()
        )
        print(f"[TRANSCRIPT] Got transcript via YouTubeTranscriptApi (length={len(text)})")
        return text
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"[TRANSCRIPT] No transcript available for {youtube_id}: {e}")
        return None
    except Exception as e:
        print(
            f"[TRANSCRIPT] YouTubeTranscriptApi failed unexpectedly: {e} "
            "– will fall back to Whisper."
        )
        return None


MAX_AUDIO_SIZE = 24 * 1024 * 1024  # 24 MB (OpenAI limit is ~25MB)


def split_audio_if_needed(path: str) -> List[str]:
    """
    If the audio file is larger than MAX_AUDIO_SIZE, split it into multiple
    smaller files with approximately equal duration. Returns a list of paths.
    """
    size = os.path.getsize(path)
    if size <= MAX_AUDIO_SIZE:
        return [path]

    print(f"[AUDIO] File {path} is {size} bytes, splitting into chunks...")

    audio = AudioSegment.from_file(path)
    duration_ms = len(audio)
    target_chunks = max(1, size // MAX_AUDIO_SIZE + 1)
    chunk_duration_ms = duration_ms // target_chunks

    base, ext = os.path.splitext(path)
    ext = ext.lstrip(".")

    chunk_paths: List[str] = []
    for i in range(target_chunks):
        start_ms = int(i * chunk_duration_ms)
        end_ms = int(min((i + 1) * chunk_duration_ms, len(audio)))
        chunk = audio[start_ms:end_ms]
        chunk_path = f"{base}_part{i}.{ext}"
        chunk.export(chunk_path, format=ext)
        chunk_paths.append(chunk_path)

    print(f"[AUDIO] Created {len(chunk_paths)} chunks.")
    return chunk_paths


def transcribe_with_whisper(path: str, model: str = "whisper-1") -> str:
    """
    Transcribe audio file with OpenAI Whisper.
    If the file is too large, split it into chunks and transcribe each part.
    """
    audio_paths = split_audio_if_needed(path)
    texts: List[str] = []
    for i, audio_path in enumerate(audio_paths):
        print(f"[WHISPER] Transcribing chunk {i + 1}/{len(audio_paths)}: {audio_path}")
        with open(audio_path, "rb") as f:
            transcript = openai_client.audio.transcriptions.create(
                model=model,
                file=f,
            )
        texts.append(transcript.text)

    full_text = "\n\n".join(texts)
    print(f"[WHISPER] Done. Combined transcript length: {len(full_text)} characters")
    return full_text
