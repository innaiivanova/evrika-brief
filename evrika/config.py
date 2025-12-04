# evrika/config.py
"""
Shared configuration and global clients for Evrika Briefs.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# Load environment variables from .env if present
load_dotenv(find_dotenv())


def get_env(name: str) -> str:
    """Read an environment variable or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Set it in your shell or in a .env file."
        )
    return value


SUPABASE_URL = get_env("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_env("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = get_env("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

# -------- Chat history (simple memory) --------
CHAT_HISTORY: List[HumanMessage] = []

# -------- "Current" YouTube video for this process / session --------
CURRENT_YOUTUBE_ID: Optional[str] = None


def set_current_youtube_id(youtube_id: str) -> None:
    """
    Remember the 'current' YouTube video for this Python process.

    RAG + metadata helpers use this when the user asks follow-up
    questions without repeating the URL or ID.
    """
    global CURRENT_YOUTUBE_ID
    CURRENT_YOUTUBE_ID = youtube_id
