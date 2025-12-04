# evrika/metadata_tool.py
"""
LangChain tool for reading *compact* video metadata from Supabase.

This lets the agent answer questions like:
- Who is the speaker / which channel is this?
- What is the title?
- How long is the video?
- When was it published?

IMPORTANT: we deliberately return ONLY a small subset of metadata
to avoid context_length_exceeded errors.
"""

import json
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from . import config
from .rag_pipeline import extract_youtube_id


def _get_video_metadata_from_supabase(youtube_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a single metadata record for the given youtube_id from the 'documents' table.

    Because each chunk shares the same video-level metadata, we only need one row.
    """
    resp = (
        config.supabase.table("documents")
        .select("metadata")
        .contains("metadata", {"youtube_id": youtube_id})
        .limit(1)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return None

    metadata = rows[0].get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    # This raw_meta can be HUGE – we will only use it to derive a few fields,
    # but we will NOT include it in the result we send to the LLM.
    raw_meta = metadata.get("raw_meta") or {}

    # ---- Compact, safe view for the LLM (NO raw_meta!) ----
    title = metadata.get("title") or raw_meta.get("title")
    url = metadata.get("url") or raw_meta.get("webpage_url")

    duration_seconds = metadata.get("duration_seconds") or raw_meta.get("duration")

    channel = (
        metadata.get("channel")
        or raw_meta.get("channel")
        or raw_meta.get("uploader")
    )

    speaker = (
        metadata.get("speaker")
        or raw_meta.get("uploader")
        or channel
    )

    published_at = metadata.get("published_at")
    if not published_at:
        upload_date = metadata.get("upload_date") or raw_meta.get("upload_date")
        if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
            published_at = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        else:
            published_at = upload_date

    # NOTE: we intentionally DO NOT include the huge raw_meta blob here.
    return {
        "youtube_id": youtube_id,
        "title": title,
        "url": url,
        "channel": channel,
        "speaker": speaker,
        "duration_seconds": duration_seconds,
        "published_at": published_at,
    }


@tool("video_metadata")
def video_metadata_tool(video_hint: str = "") -> str:
    """
    Get metadata for a YouTube video (title, speaker, channel, duration, publish date, URL).

    - video_hint: optional YouTube URL or ID. If omitted, uses the most recently
      ingested / referenced video in this session.

    Returns a SMALL JSON string with keys:
      - youtube_id
      - title
      - url
      - channel
      - speaker
      - duration_seconds
      - published_at
    """
    youtube_id: Optional[str] = None

    # 1) Try to parse explicit hint if provided
    if video_hint:
        try:
            youtube_id = extract_youtube_id(video_hint)
        except Exception as e:
            return f"Could not extract a YouTube ID from video_hint={video_hint!r}: {e}"

    # 2) Fall back to the 'current' video if we have one
    if not youtube_id:
        youtube_id = config.CURRENT_YOUTUBE_ID

    if not youtube_id:
        return (
            "I don't know which video you mean. "
            "Please either provide a YouTube URL/ID or ingest a video first using fetch_video."
        )

    meta_view = _get_video_metadata_from_supabase(youtube_id)
    if not meta_view:
        return f"No metadata found in Supabase for youtube_id={youtube_id}."

    # Keep CURRENT_YOUTUBE_ID in sync
    config.set_current_youtube_id(youtube_id)

    # This JSON is small enough to safely send to the LLM
    return json.dumps(meta_view, default=str)
