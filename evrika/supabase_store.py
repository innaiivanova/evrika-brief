# evrika/supabase_store.py
"""
Helpers for storing and retrieving documents in Supabase (pgvector).
"""

from typing import List
from uuid import uuid4

from langchain_core.documents import Document

from .config import supabase, embeddings


def get_existing_chunk_count(youtube_id: str) -> int:
    """
    Check Supabase 'documents' table for any rows whose metadata->>youtube_id
    matches the given youtube_id.
    """
    try:
        response = (
            supabase.table("documents")
            .select("id")
            .eq("metadata->>youtube_id", youtube_id)
            .execute()
        )
        data = response.data or []
        count = len(data)
        if count > 0:
            print(f"[INGEST] Found {count} existing chunks in Supabase for youtube_id={youtube_id}")
        return count
    except Exception as e:
        print(f"[INGEST] Warning: failed to check existing chunks ({e}), assuming none.")
        return 0


def store_docs_in_supabase(docs: List[Document]) -> None:
    """
    Store document chunks + embeddings in Supabase `documents` table.
    Uses pgvector for the embedding column.
    """
    try:
        texts = [d.page_content for d in docs]
        vectors = embeddings.embed_documents(texts)

        rows = []
        for doc, vec in zip(docs, vectors):
            rows.append(
                {
                    "id": str(uuid4()),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": vec,
                }
            )

        supabase.table("documents").insert(rows).execute()
        print(f"[SUPABASE] Stored {len(rows)} chunks in 'documents' table.")
    except Exception as e:
        print(f"[SUPABASE] Warning: failed to store docs in Supabase: {e}")


def retrieve_docs_from_supabase(query: str, k: int = 10) -> List[Document]:
    """
    Use Supabase RPC `match_documents` to retrieve k nearest chunks for a query.
    Expects an RPC function `match_documents` in your database.
    """
    try:
        embedding = embeddings.embed_query(query)

        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": embedding,
                "match_count": k,
            },
        ).execute()

        rows = response.data or []
        docs: List[Document] = []
        for row in rows:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata=row.get("metadata") or {},
                )
            )
        print(f"[SUPABASE] Retrieved {len(docs)} docs from match_documents.")
        return docs
    except Exception as e:
        print(f"[SUPABASE] Error in retrieve_docs_from_supabase: {e}")
        return []
