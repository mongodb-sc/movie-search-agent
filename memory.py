"""
Long-term memory for the movie agent, stored in the same MongoDB Atlas cluster.
Short-term (conversation) memory is handled by LangGraph's MongoDB checkpointer.
"""
import os
from datetime import datetime
from typing import Any

from tools import get_mongo_client

# Database and collection for long-term memory (same cluster as sample_mflix)
MEMORY_DB_NAME = os.environ.get("AGENT_MEMORY_DB", "agent_memory")
MEMORY_COLLECTION = "long_term_memory"


def get_long_term_memory(thread_id: str, limit: int = 20) -> str:
    """
    Load long-term memory entries for this thread from MongoDB.
    Returns a formatted string for inclusion in the system prompt.
    """
    try:
        client = get_mongo_client()
        coll = client[MEMORY_DB_NAME][MEMORY_COLLECTION]
        cursor = coll.find(
            {"thread_id": thread_id},
            {"content": 1, "created_at": 1, "_id": 0},
        ).sort("created_at", -1).limit(limit)
        rows = list(cursor)
    except Exception:
        return ""
    if not rows:
        return "(none yet)"
    lines = []
    for r in rows:
        content = (r.get("content") or "").strip()
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines) if lines else "(none yet)"


def add_long_term_memory(thread_id: str, content: str) -> None:
    """Store a long-term memory entry for this thread."""
    if not (content or "").strip():
        return
    client = get_mongo_client()
    coll = client[MEMORY_DB_NAME][MEMORY_COLLECTION]
    coll.insert_one({
        "thread_id": thread_id,
        "content": content.strip(),
        "created_at": datetime.utcnow(),
    })
