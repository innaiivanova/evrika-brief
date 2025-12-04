# evrika/agent.py
"""
Simple LangChain tool-calling agent for Evrika Briefs.
"""

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from .config import llm, CHAT_HISTORY
from .rag_pipeline import (
    fetch_video,
    semantic_search,
    video_chat,
    generate_brief,
    recommendations,
    save_brief_as_pdf,
)
from .metadata_tool import video_metadata_tool


# -------- Tools --------

TOOLS = [
    fetch_video,
    semantic_search,
    video_chat,
    generate_brief,
    recommendations,
    save_brief_as_pdf,
    video_metadata_tool,  # metadata tool is available to the LLM
]
TOOLS_BY_NAME: Dict[str, Any] = {t.name: t for t in TOOLS}

llm_with_tools = llm.bind_tools(TOOLS)


# -------- System Prompt --------

SYSTEM_PROMPT = """
You are Evrika Briefs, an assistant that works with YouTube videos.

LANGUAGE

- You must ALWAYS respond in English, regardless of the language used in the
  user's message or in any tool output.
- If the user writes in another language, understand it, but respond only in English.

TOOLS YOU CAN USE

- fetch_video:
    Ingest a YouTube video by URL or ID into the system.
- semantic_search:
    Retrieve the most relevant transcript chunks for a question.
- video_chat:
    Answer questions about the CONTENT of the video using transcript chunks.
    Use this for questions about ideas, explanations, concepts, arguments, etc.
- generate_brief:
    Create a 1-page Evrika Brief summary for a video.
- recommendations:
    Suggest follow-up learning directions after a video.
- save_brief_as_pdf:
    Save a given brief as a PDF file.
- video_metadata:
    Get structured METADATA for a video from Supabase, including:
    title, speaker, channel, duration, publish date, and URL.

GUIDELINES

- Use `video_chat` for questions about the content of the video
  (concepts, explanations, insights, quotes, etc.).
- Use `video_metadata` for questions about metadata
  (title, speaker, channel, duration, publish date, URL).
- If a tool returns JSON, read it carefully and answer based on it.
- Keep answers concise and helpful.
""".strip()


# -------- Limits to avoid context blowups --------

MAX_HISTORY_MESSAGES = 12          # number of previous messages to keep
MAX_TOOL_OUTPUT_CHARS = 40_000     # hard cap for any single tool output


def _sanitize_tool_output(tool_name: str, tool_output: Any) -> str:
    """
    Convert tool_output to a reasonably sized string before adding it
    as a ToolMessage.

    - Coerces non-strings to string.
    - If it's JSON, drop obviously huge keys like 'raw_meta' / 'raw_metadata'.
    - Hard-caps the final length to MAX_TOOL_OUTPUT_CHARS.
    """
    if not isinstance(tool_output, str):
        tool_output = str(tool_output)

    # Fast path if already small
    if len(tool_output) <= MAX_TOOL_OUTPUT_CHARS:
        return tool_output

    # Try JSON cleanup first (useful for metadata)
    try:
        data = json.loads(tool_output)
        if isinstance(data, dict):
            if "raw_meta" in data:
                data["raw_meta"] = "[omitted: raw_meta too large]"
            if "raw_metadata" in data:
                data["raw_metadata"] = "[omitted: raw_metadata too large]"
        tool_output = json.dumps(data)
    except Exception:
        # Not JSON or parse failed; we'll just truncate below
        pass

    # Hard cap to avoid context_length_exceeded
    if len(tool_output) > MAX_TOOL_OUTPUT_CHARS:
        tool_output = tool_output[:MAX_TOOL_OUTPUT_CHARS] + "... [truncated]"

    return tool_output


# -------- Main agent loop --------

def agent_respond(user_input: str) -> str:
    """
    Simple LangChain-based tool-calling loop.

    The LLM sees the system prompt and the available tools, decides
    which tools to call (if any), and we execute them in a loop until
    the LLM returns a normal message with no tool calls.
    """
    # Only include the last N messages from history to keep context small
    history = CHAT_HISTORY[-MAX_HISTORY_MESSAGES:]

    messages: List = [SystemMessage(content=SYSTEM_PROMPT)] + history + [
        HumanMessage(user_input)
    ]

    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # No tool calls -> final answer
        if not getattr(ai_msg, "tool_calls", None):
            # Persist the latest exchange (without the system message)
            CHAT_HISTORY.extend([HumanMessage(user_input), ai_msg])

            # Keep global history bounded as well
            if len(CHAT_HISTORY) > MAX_HISTORY_MESSAGES:
                CHAT_HISTORY[:] = CHAT_HISTORY[-MAX_HISTORY_MESSAGES:]

            return ai_msg.content

        # Execute each requested tool and feed results back to the model
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            call_id = tool_call["id"]

            tool_obj = TOOLS_BY_NAME.get(tool_name)
            if tool_obj is None:
                raw_output = f"Error: unknown tool '{tool_name}'."
            else:
                raw_output = tool_obj.invoke(tool_args)

            safe_output = _sanitize_tool_output(tool_name, raw_output)

            messages.append(
                ToolMessage(
                    content=safe_output,
                    tool_call_id=call_id,
                )
            )


if __name__ == "__main__":
    print(
        "Evrika Briefs Agent - LangChain tools + Supabase pgvector "
        "(YouTubeTranscriptApi + yt-dlp + Whisper)"
    )
    print("Type a message (or 'exit' to quit).")
    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        try:
            answer = agent_respond(user_input)
            print("\nAgent:", answer)
        except Exception as e:
            print(f"\nError while running agent: {e}")
