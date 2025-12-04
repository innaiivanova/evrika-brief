"""
RAG helpers and LangChain tools for Evrika Briefs.

This module handles:
- Ingesting YouTube videos into Supabase (YouTubeTranscriptApi + Whisper).
- Semantic search over stored chunks.
- Chatting with a video (RAG).
- Generating a one-page "Evrika Brief" in a fixed template.
- Simple recommendations.
- Saving a brief as a PDF.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from langchain_core.tools import tool

from .config import (
    supabase,
    embeddings,
    llm,
    set_current_youtube_id,
    CURRENT_YOUTUBE_ID,
)
from .transcripts import (
    fetch_metadata_with_ytdlp,
    fetch_audio_with_ytdlp,
    try_fetch_transcript_via_api,
    transcribe_with_whisper,
)

# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------


def extract_youtube_id(url_or_id: str) -> str:
    """
    Extract a YouTube video ID from either:
    - a plain 11-character ID
    - a full YouTube URL
    - a youtu.be short URL
    """
    if not url_or_id:
        raise ValueError("Empty YouTube URL/ID")

    # If it looks like a bare ID (11 chars, no slash), accept it
    if len(url_or_id) == 11 and "/" not in url_or_id and "?" not in url_or_id:
        return url_or_id

    parsed = urlparse(url_or_id)

    # youtu.be/<id>
    if parsed.hostname in {"youtu.be"}:
        vid = parsed.path.lstrip("/")
        if vid:
            return vid

    # youtube.com/watch?v=<id>
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]

    # Fallback: last path segment
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        return path_parts[-1]

    raise ValueError(f"Could not extract YouTube ID from: {url_or_id}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Simple word-based text chunker.

    - chunk_size: approximate number of words per chunk.
    - overlap: number of words to overlap between consecutive chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        # move forward with overlap
        start += max(1, chunk_size - overlap)

    return chunks


def _existing_chunk_count(youtube_id: str) -> int:
    """
    Check how many chunks we already have stored in Supabase for this video.
    """
    resp = (
        supabase.table("documents")
        .select("id")
        .contains("metadata", {"youtube_id": youtube_id})
        .execute()
    )
    data = resp.data or []
    count = len(data)
    if count:
        print(f"[INGEST] Found {count} existing chunks in Supabase for youtube_id={youtube_id}")
    return count


# ---------------------------------------------------------------------------
# METADATA HELPERS
# ---------------------------------------------------------------------------


def _fetch_metadata_row(youtube_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a single metadata record for the given youtube_id from 'documents'.
    """
    resp = (
        supabase.table("documents")
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
    return metadata


def _build_metadata_view(youtube_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the stored metadata into a compact structure for the LLM.
    """
    raw_meta = metadata.get("raw_meta") or {}

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
        or raw_meta.get("speaker")
        or raw_meta.get("artist")
    )

    if speaker and channel and speaker == channel:
        speaker = None

    published_at = metadata.get("published_at")
    if not published_at:
        upload_date = metadata.get("upload_date") or raw_meta.get("upload_date")
        if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
            published_at = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        else:
            published_at = upload_date

    return {
        "youtube_id": youtube_id,
        "title": title,
        "url": url,
        "channel": channel,
        "speaker": speaker,
        "duration_seconds": duration_seconds,
        "published_at": published_at,
    }


def _is_metadata_question(question: str) -> bool:
    q = question.lower().strip()
    core_keywords = [
        "title of the video",
        "video title",
        "name of the video",
        "what is the title",
        "what's the title",
        "who is the speaker",
        "who's the speaker",
        "who is speaking",
        "who is the host",
        "how long is the video",
        "how long is it",
        "what is the duration",
        "video duration",
        "when was this video published",
        "when was it published",
        "when was this uploaded",
        "upload date",
        "publish date",
    ]
    if any(k in q for k in core_keywords):
        return True
    if "channel" in q:
        return True
    if "url" in q or "link" in q:
        return True
    return False


def _is_recommendation_question(question: str) -> bool:
    q = question.lower()
    explicit_patterns = [
        "recommend me similar videos",
        "recommend similar videos",
        "similar videos",
        "similar video",
        "what should i watch next",
        "what else should i watch",
        "follow-up videos",
        "related videos",
        "more like this",
        "next video to watch",
    ]
    if any(p in q for p in explicit_patterns):
        return True
    if "recommend" in q and ("video" in q or "watch" in q or "content" in q):
        return True
    return False


def _answer_recommendation_question(question: str, youtube_id: Optional[str]) -> str:
    if not youtube_id:
        return (
            "I don't know which video to base recommendations on. "
            "Please provide a YouTube URL or ingest a video first."
        )

    docs = _get_all_chunks_for_video(youtube_id)
    if not docs:
        return (
            "I couldn't find any stored chunks for this video, "
            "so I can't generate recommendations yet."
        )

    context = "\n\n".join(doc["content"] for doc in docs[:5])

    prompt = f"""
You are a learning coach inside Evrika Briefs.

A user has just watched a YouTube video (id={youtube_id}) and asked:

{question}

Here are a few chunks from the video transcript for context:
\"\"\"{context}\"\"\"

Suggest 3–7 concrete follow-up learning steps, including:
- Search queries they could type into YouTube or Google
- Concrete topics or subskills to explore next
- Optional: types of videos (tutorial, case study, lecture, etc.)

Return the answer as a Markdown bullet list.
""".strip()

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def _answer_metadata_question(question: str, youtube_id: Optional[str]) -> str:
    if not youtube_id:
        return (
            "I don't know which video you mean. Please either ingest a video first "
            "or provide a specific YouTube URL or ID."
        )

    metadata_row = _fetch_metadata_row(youtube_id)
    if not metadata_row:
        return f"I couldn't find stored metadata for this video (id={youtube_id})."

    view = _build_metadata_view(youtube_id, metadata_row)
    metadata_json = json.dumps(view, indent=2)

    prompt = f"""
You are Evrika Briefs. The user asked a question about video METADATA
(not about the content / ideas of the video).

Here is the stored metadata for the video, as JSON:
{metadata_json}

User question:
{question}

Answer the question using ONLY this metadata. If something is missing,
say you are not sure. Answer in 1–3 concise sentences.
""".strip()

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# ---------------------------------------------------------------------------
# CHUNK STORAGE
# ---------------------------------------------------------------------------


def _store_chunks_and_embeddings(
    youtube_id: str,
    title: str,
    url: str,
    chunks: List[str],
    raw_meta: Optional[Dict[str, Any]] = None,
) -> int:
    if not chunks:
        print("[INGEST] No chunks to store.")
        return 0

    duration_seconds: Optional[float] = None
    channel: Optional[str] = None
    upload_date: Optional[str] = None
    published_at: Optional[str] = None

    if raw_meta:
        duration_seconds = raw_meta.get("duration")
        channel = raw_meta.get("channel") or raw_meta.get("uploader")
        upload_date = raw_meta.get("upload_date")
        if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
            published_at = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        else:
            published_at = upload_date

    base_metadata: Dict[str, Any] = {
        "title": title,
        "url": url,
        "youtube_id": youtube_id,
        "source": "evrika-briefs",
        "duration_seconds": duration_seconds,
        "channel": channel,
        "published_at": published_at,
        "raw_meta": raw_meta or {},
    }

    print(f"[INGEST] Embedding {len(chunks)} chunks...")
    vectors = embeddings.embed_documents(chunks)

    rows: List[Dict[str, Any]] = []
    for content, embedding in zip(chunks, vectors):
        rows.append(
            {
                "content": content,
                "metadata": base_metadata,
                "embedding": embedding,
            }
        )

    supabase.table("documents").insert(rows).execute()
    print(f"[INGEST] Stored {len(rows)} chunks in Supabase.")
    return len(rows)


# ---------------------------------------------------------------------------
# RAG HELPERS
# ---------------------------------------------------------------------------


def _get_all_chunks_for_video(youtube_id: str) -> List[Dict[str, Any]]:
    resp = (
        supabase.table("documents")
        .select("id, content, metadata")
        .contains("metadata", {"youtube_id": youtube_id})
        .execute()
    )
    data = resp.data or []
    print(f"[SUPABASE] _get_all_chunks_for_video: loaded {len(data)} chunks for {youtube_id}")
    return data


def _match_documents(
    query: str,
    youtube_id: Optional[str] = None,
    match_count: int = 6,
) -> List[Dict[str, Any]]:
    query_embedding = embeddings.embed_query(query)

    initial_match_count = match_count if youtube_id is None else max(
        match_count * 3, match_count + 10
    )

    payload: Dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": initial_match_count,
    }

    resp = supabase.rpc("match_documents", payload).execute()
    docs = resp.data or []

    if youtube_id:
        filtered: List[Dict[str, Any]] = []
        for d in docs:
            meta = d.get("metadata") or {}
            if meta.get("youtube_id") == youtube_id:
                filtered.append(d)
        docs = filtered

    docs = docs[:match_count]

    print(f"[SUPABASE] Retrieved {len(docs)} docs from match_documents (after filtering).")
    return docs


# ---------------------------------------------------------------------------
# INGESTION
# ---------------------------------------------------------------------------


def ingest_youtube(url: str) -> Dict[str, Any]:
    """
    Ingest a YouTube video into Supabase if not already present.
    """
    youtube_id = extract_youtube_id(url)

    existing_count = _existing_chunk_count(youtube_id)
    if existing_count:
        print("[INGEST] Skipping ingestion; already found chunks.")
        set_current_youtube_id(youtube_id)
        return {
            "title": "",
            "youtube_id": youtube_id,
            "chunk_count": existing_count,
        }

    print(f"[INGEST] Fetching metadata for {url}")
    meta = fetch_metadata_with_ytdlp(url)
    youtube_id = meta.get("id", youtube_id)
    title = meta.get("title", "")

    transcript_text = try_fetch_transcript_via_api(youtube_id)

    if not transcript_text:
        print("[INGEST] Falling back to Whisper on audio via yt-dlp...")
        audio_path, meta = fetch_audio_with_ytdlp(url)
        transcript_text = transcribe_with_whisper(audio_path)

    chunks = chunk_text(transcript_text)
    chunk_count = _store_chunks_and_embeddings(
        youtube_id=youtube_id,
        title=title,
        url=meta.get("webpage_url", url),
        chunks=chunks,
        raw_meta=meta,
    )

    set_current_youtube_id(youtube_id)

    print(f"[INGEST] Completed ingestion for youtube_id={youtube_id}")
    return {
        "title": title,
        "youtube_id": youtube_id,
        "chunk_count": chunk_count,
    }


# ---------------------------------------------------------------------------
# LANGCHAIN TOOLS + QA
# ---------------------------------------------------------------------------


@tool
def fetch_video(url: str) -> str:
    """
    Ingest a YouTube video into Evrika Briefs.
    """
    meta = ingest_youtube(url)
    title = meta.get("title") or "(title unknown)"
    youtube_id = meta.get("youtube_id", "")
    chunk_count = meta.get("chunk_count", 0)
    return (
        f"Ingested video '{title}' (id={youtube_id}) into Evrika Briefs with "
        f"{chunk_count} chunks."
    )


@tool
def semantic_search(question: str, video_hint: str = "") -> str:
    """
    Search the ingested video semantically and return the top matching chunks.
    """
    youtube_id: Optional[str] = None
    if video_hint:
        try:
            youtube_id = extract_youtube_id(video_hint)
            set_current_youtube_id(youtube_id)
        except Exception as e:
            print(f"[SEMANTIC_SEARCH] Failed to parse video_hint '{video_hint}': {e}")

    docs = _match_documents(question, youtube_id=youtube_id, match_count=6)
    if not docs:
        return "No matching chunks found in the knowledge base."

    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        formatted_chunks.append(f"[Chunk {i}]\n{doc.get('content', '')}")

    return "\n\n---\n\n".join(formatted_chunks)


def _run_qa(question: str, video_hint: str = "") -> str:
    youtube_id: Optional[str] = None
    if video_hint:
        try:
            youtube_id = extract_youtube_id(video_hint)
            set_current_youtube_id(youtube_id)
        except Exception as e:
            print(f"[VIDEO_CHAT] Failed to parse video_hint '{video_hint}': {e}")
    else:
        youtube_id = CURRENT_YOUTUBE_ID

    if _is_metadata_question(question):
        return _answer_metadata_question(question, youtube_id)

    if _is_recommendation_question(question):
        return _answer_recommendation_question(question, youtube_id)

    docs = _match_documents(question, youtube_id=youtube_id, match_count=6)

    if (not docs) and youtube_id:
        print(
            "[QA] match_documents returned 0 docs; "
            "falling back to all chunks for this video."
        )
        docs = _get_all_chunks_for_video(youtube_id)

    if not docs:
        return "No matching chunks found in the knowledge base."

    context = "\n\n".join(
        f"[Chunk {i}] {doc.get('content', '')}"
        for i, doc in enumerate(docs, start=1)
    )

    prompt = f"""
You are Evrika Briefs, an assistant that answers questions about YouTube videos
using their transcript chunks.

Use ONLY the information from the provided chunks to answer the question.
If the answer is not clearly in the chunks, say that you are not sure.

Relevant chunks:
{context}

User question:
{question}

Answer in a clear, concise way, 3–7 sentences maximum.
""".strip()

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


@tool
def video_chat(question: str, video_hint: str = "") -> str:
    """
    Answer a question about an ingested YouTube video using RAG.
    """
    return _run_qa(question, video_hint)


def answer_question_text(question: str, video_hint: str = "") -> str:
    """
    Plain helper for answering a question about the ingested videos (for HTTP APIs).
    """
    return _run_qa(question, video_hint)


# ---------------------------------------------------------------------------
# BRIEF GENERATION
# ---------------------------------------------------------------------------


def generate_brief_text(video_hint: str) -> str:
    """
    Generate a structured Evrika Brief in Markdown for the given YouTube URL or ID.
    """
    youtube_id = extract_youtube_id(video_hint)
    docs = _get_all_chunks_for_video(youtube_id)
    if not docs:
        ingest_youtube(video_hint)
        docs = _get_all_chunks_for_video(youtube_id)
        if not docs:
            return "I could not find or ingest this video to generate a brief."

    full_text = "\n\n".join(doc["content"] for doc in docs)

    metadata_row = _fetch_metadata_row(youtube_id) or {}
    view = _build_metadata_view(youtube_id, metadata_row)

    video_title = view.get("title") or "(title unknown)"
    video_url = view.get("url") or f"https://youtu.be/{youtube_id}"
    channel = view.get("channel")
    speaker = view.get("speaker")

    if channel:
        creator_line = f"- Creator / Channel: {channel}"
    else:
        creator_line = "- Creator / Channel: Unknown"

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    generated_line = f"Generated: {generated_at}"

    prompt = f"""
You are Evrika Briefs, a tool that turns YouTube videos into smart 1-page briefs.

You generate markdown that will be exported to PDF.

Very important formatting rules:

1. Use EXACTLY the headings and section order from the template.
2. Do NOT add new sections or remove any sections.
3. Do NOT add any explanatory text like “leave empty for the user to fill in”.
4. For the section "## Personal Notes":
   - Do NOT change it at all.
   - Leave the three bullet lines exactly as they are: "- …" on each line.
5. Keep the line starting with "Generated:" directly under the title. If you don't know the time, write "Generated: Unknown".
6. For "Source, Links & References":
   - Fill only the bracketed or clearly labeled parts (title, URL, creator), but do not add extra commentary paragraphs.
7. If information is unknown, write "Unknown" ONLY in clearly labeled places (e.g. "Creator / Channel: Unknown").
8. Keep the markdown syntax valid and do not escape or reformat the template structure.

Now, using the template below, replace only the placeholder text in square brackets or obviously placeholder bullet items ("model 1", "insight 1", etc.) with content derived from the video. For "The Main Idea", replace the placeholder line with your own 5–8 sentence summary (no bullet points).

Write a structured brief in Markdown following exactly this structure:

# [Evrika Brief - Video Title (Speaker)]

Generated: Unknown

## The Main Idea

What is this really about? Why does it matter? Summarize in up to 5–8 short sentences.

---

## Relevant Models & Frameworks

- [model 1] – short description, when/why to use.
- [model 2] – short description, when/why to use.
- [model 3] – short description, when/why to use.
- [model 4] – if relevant.
- [model 5] – if relevant.

---

## Top Insights

- [insight 1]
- [insight 2]
- [insight 3]
- [insight 4] (if needed)
- [insight 5] (if needed)

---

## Memorable Quotes

- “Quote 1…” — [who said it, if relevant]
- “Quote 2…” — [who said it, if relevant]
- “Quote 3…” — [who said it, if relevant]

---

## How to Apply This

- 3–5 short, practical suggestions

---

## Personal Notes

- …
- …
- …

---

## Source, Links & References

- Original video (title + URL): [Video Title] – [Video URL]
- Creator / Channel:
- Resources or tools explicitly mentioned in the video:

Base your brief ONLY on the transcript text below.
If you are unsure about specific names or numbers, be honest and approximate.

TRANSCRIPT:
\"\"\"{full_text}\"\"\"

Now write the Evrika Brief:
""".strip()

    response = llm.invoke(prompt)
    brief = getattr(response, "content", str(response))

    # Post-process header, Generated line, Source, Creator
    if speaker:
        header_line = f"# Evrika Brief - {video_title} ({speaker})"
    else:
        header_line = f"# Evrika Brief - {video_title}"

    source_line = f"- Original video (title + URL): {video_title} – {video_url}"

    lines = brief.splitlines()
    header_index: Optional[int] = None
    generated_found = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# Evrika Brief -"):
            header_index = i
            lines[i] = header_line
        elif stripped.startswith("Generated:") and not generated_found:
            lines[i] = generated_line
            generated_found = True
        elif stripped.startswith("- Original video (title + URL):"):
            lines[i] = source_line
        elif stripped.startswith("- Creator / Channel:"):
            lines[i] = creator_line

    if not generated_found and header_index is not None:
        lines.insert(header_index + 1, generated_line)

    brief = "\n".join(lines)
    return brief


@tool
def generate_brief(video_hint: str) -> str:
    """
    LangChain Tool wrapper around generate_brief_text so agents can still call it.
    """
    return generate_brief_text(video_hint)


@tool
def recommendations(video_hint: str, learning_goal: str = "") -> str:
    """
    Suggest follow-up learning directions or search ideas based on a video.
    """
    youtube_id = extract_youtube_id(video_hint)
    docs = _get_all_chunks_for_video(youtube_id)
    context = "\n\n".join(doc["content"] for doc in docs[:5])

    prompt = f"""
You are a learning coach inside Evrika Briefs.

A user has just watched a YouTube video and ingested it into the system.
They may have the following learning goal (optional):

Learning goal: {learning_goal or "(none given)"}

Here are a few chunks from the video transcript for context:
\"\"\"{context}\"\"\"

Suggest 3–7 concrete follow-up learning steps, including:
- Search queries they could type into YouTube or Google
- Concrete topics or subskills to explore next
- Optional: types of videos (tutorial, case study, lecture, etc.)

Return the answer as a Markdown bullet list.
""".strip()

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# ---------------------------------------------------------------------------
# PDF GENERATION (NO TOOL)
# ---------------------------------------------------------------------------

def save_brief_as_pdf(brief_text: str, filename: str = "evrika_brief.pdf") -> str:
    """
    Turn a Markdown-ish Evrika Brief into a styled PDF and return the filename.

    - Main title: font 18, bold, with proper line spacing.
    - Section headings: font 14, bold, with extra space before/after.
    - Body text & bullets: font 12, with comfortable line spacing.
    - '---' becomes a visible horizontal line with padding.
    - Inline **bold** is rendered in bold.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return (
            "The 'reportlab' package is not installed on the server environment. "
            "Install it with 'pip install reportlab' to generate PDFs."
        )

    # Create canvas
    c = canvas.Canvas(filename, pagesize=letter)
    page_width, page_height = letter

    # Page margins
    margin_left = 50
    margin_right = 50
    top_margin = 60
    bottom_margin = 60

    max_width = page_width - margin_left - margin_right

    # Base spacing unit (used for extra padding between blocks)
    base_line_height = 14

    # Current y position
    y = page_height - top_margin

    def new_page() -> float:
        """Start a new page and return the new y position."""
        c.showPage()
        return page_height - top_margin

    def draw_line_with_bold(
        line_text: str,
        y_pos: float,
        font_name: str,
        font_size: int,
    ) -> None:
        """
        Draw a single line of text at (margin_left, y_pos),
        interpreting Markdown-style **bold** segments.
        """
        if "**" not in line_text:
            c.setFont(font_name, font_size)
            c.drawString(margin_left, y_pos, line_text)
            return

        parts = line_text.split("**")
        # parts alternate: normal, bold, normal, bold, ...
        is_bold = False
        x = margin_left

        for part in parts:
            if part == "":
                # skip empty pieces (e.g. "****")
                is_bold = not is_bold
                continue

            if is_bold:
                # draw bold segment
                c.setFont("Helvetica-Bold", font_size)
                c.drawString(x, y_pos, part)
                x += c.stringWidth(part, "Helvetica-Bold", font_size)
            else:
                # draw normal segment
                c.setFont(font_name, font_size)
                c.drawString(x, y_pos, part)
                x += c.stringWidth(part, font_name, font_size)

            is_bold = not is_bold

    def draw_wrapped(
        text: str,
        y_pos: float,
        font_name: str = "Helvetica",
        font_size: int = 10,
        extra_space_after: float = 0.0,
        support_inline_bold: bool = False,
    ) -> float:
        """
        Draw a block of text with word-wrapping within left/right margins.
        Adds extra vertical space after the block.

        If support_inline_bold=True, treat **...** segments as bold.
        """
        c.setFont(font_name, font_size)
        words = text.split()
        if not words:
            return y_pos

        line = ""
        # Line height scales with font size → more space between title lines
        line_height = int(font_size * 1.4)

        for word in words:
            candidate = (line + " " + word).strip()
            # Measure width ignoring bold; good enough for wrapping
            if c.stringWidth(candidate, font_name, font_size) <= max_width:
                line = candidate
            else:
                # Draw current line
                if support_inline_bold:
                    draw_line_with_bold(line, y_pos, font_name, font_size)
                else:
                    c.drawString(margin_left, y_pos, line)
                y_pos -= line_height
                if y_pos < bottom_margin:
                    y_pos = new_page()
                    c.setFont(font_name, font_size)
                line = word

        if line:
            if support_inline_bold:
                draw_line_with_bold(line, y_pos, font_name, font_size)
            else:
                c.drawString(margin_left, y_pos, line)
            y_pos -= line_height
            if y_pos < bottom_margin:
                y_pos = new_page()
                c.setFont(font_name, font_size)

        # Extra spacing after the whole block
        y_pos -= extra_space_after
        return y_pos

    first_header_done = False  # to detect main title

    for raw_line in brief_text.split("\n"):
        stripped = raw_line.strip()

        # Completely blank line → vertical space
        if not stripped:
            y -= base_line_height * 0.75
            if y < bottom_margin:
                y = new_page()
            continue

        # Horizontal rule: a line that is only dashes (--- etc.)
        if all(ch in "-–—" for ch in stripped) and len(stripped) >= 3:
            # space before line
            y -= base_line_height * 0.5
            if y < bottom_margin:
                y = new_page()

            # draw the line
            c.line(margin_left, y, page_width - margin_right, y)

            # space after line
            y -= base_line_height
            if y < bottom_margin:
                y = new_page()
            continue

        # Markdown headings: lines starting with '#'
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped[level:].strip()

            if not first_header_done:
                # MAIN TITLE – font 18, bold, extra space after.
                # Line spacing handled in draw_wrapped (font_size=18).
                y = draw_wrapped(
                    heading_text,
                    y,
                    font_name="Helvetica-Bold",
                    font_size=18,
                    extra_space_after=base_line_height * 1.0,
                    support_inline_bold=False,
                )
                first_header_done = True
            else:
                # SECTION HEADINGS – font 14, bold, with space before/after
                y -= base_line_height * 1.25
                if y < bottom_margin:
                    y = new_page()
                y = draw_wrapped(
                    heading_text,
                    y,
                    font_name="Helvetica-Bold",
                    font_size=14,
                    extra_space_after=base_line_height * 0.75,
                    support_inline_bold=False,
                )
            continue

        # "Generated: ..." line – smaller font, under title with gap after
        if stripped.startswith("Generated:"):
            y = draw_wrapped(
                raw_line,
                y,
                font_name="Helvetica",
                font_size=10,
                extra_space_after=base_line_height * 1.0,
                support_inline_bold=False,
            )
            continue

        # Bullet lines (start with "- ") – body font 12, support bold
        if stripped.startswith("- "):
            y = draw_wrapped(
                raw_line,
                y,
                font_name="Helvetica",
                font_size=12,
                extra_space_after=base_line_height * 0.3,
                support_inline_bold=True,
            )
            continue

        # Normal paragraph text – body font 12, support bold
        y = draw_wrapped(
            raw_line,
            y,
            font_name="Helvetica",
            font_size=12,
            extra_space_after=base_line_height * 0.5,
            support_inline_bold=True,
        )

    c.save()
    return filename

