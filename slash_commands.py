"""
Direct slash commands that invoke tools without an LLM round-trip.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from memory import add_long_term_memory, get_long_term_memory
from tools import get_mongo_client, movies_like, recommend_movie, semantic_search_plots

HELP_TEXT = """**Slash commands** (run tools directly, no LLM):

| Command | Example |
|---------|---------|
| `/help` | Show this list |
| `/recommend [genre] [--year-min N] [--year-max N] [--limit N]` | `/recommend sci-fi --year-min 1990 --year-max 1999` |
| `/search <query> [--limit N] [--no-rerank]` | `/search time travel paradox` |
| `/like <title> [--limit N]` | `/like "The Matrix"` |
| `/remember <text>` | `/remember I love thrillers` |
| `/memory` | Show saved facts for this session |
| `/count` | Count movies in the database |
| `/genres [--limit N]` | Top genres by movie count |
"""


@dataclass
class SlashCommandResult:
    response: str
    thinking: list[str] = field(default_factory=list)
    error: str | None = None


def is_slash_command(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("/") and len(stripped) > 1


def _split(text: str) -> tuple[str, list[str]]:
    parts = shlex.split(text.strip())
    if not parts:
        raise ValueError("Empty command.")
    name = parts[0].lstrip("/").lower()
    return name, parts[1:]


def _pop_flag(args: list[str], flag: str) -> bool:
    if flag in args:
        args.remove(flag)
        return True
    return False


def _pop_value(args: list[str], flag: str, cast=str):
    if flag not in args:
        return None
    i = args.index(flag)
    if i + 1 >= len(args):
        raise ValueError(f"{flag} requires a value.")
    value = cast(args[i + 1])
    del args[i : i + 2]
    return value


def _thinking(tool: str, **kwargs) -> list[str]:
    parts = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)
    return [f"[direct] → {tool}({parts})"]


def run_slash_command(text: str, thread_id: str = "default") -> SlashCommandResult | None:
    """Parse and run a slash command. Returns None if input is not a slash command."""
    if not is_slash_command(text):
        return None

    try:
        name, args = _split(text)
    except ValueError as e:
        return SlashCommandResult(response="", error=str(e))

    thread_id = (thread_id or "default").strip() or "default"

    try:
        if name in ("help", "?"):
            return SlashCommandResult(response=HELP_TEXT)

        if name == "recommend":
            year_min = _pop_value(args, "--year-min", int)
            year_max = _pop_value(args, "--year-max", int)
            limit = _pop_value(args, "--limit", int) or 5
            genre = " ".join(args).strip() or None
            kwargs = {"genre": genre, "year_min": year_min, "year_max": year_max, "limit": limit}
            response = recommend_movie(**kwargs)
            return SlashCommandResult(response=response, thinking=_thinking("recommend_movie", **kwargs))

        if name == "search":
            use_reranker = not _pop_flag(args, "--no-rerank")
            limit = _pop_value(args, "--limit", int) or 5
            query = " ".join(args).strip()
            if not query:
                raise ValueError("/search requires a query, e.g. `/search time travel`.")
            kwargs = {"query": query, "limit": limit, "use_reranker": use_reranker}
            response = semantic_search_plots(**kwargs)
            return SlashCommandResult(response=response, thinking=_thinking("semantic_search_plots", **kwargs))

        if name == "like":
            limit = _pop_value(args, "--limit", int) or 5
            title = " ".join(args).strip()
            if not title:
                raise ValueError("/like requires a movie title, e.g. `/like Inception`.")
            kwargs = {"movie_title": title, "limit": limit}
            response = movies_like(**kwargs)
            return SlashCommandResult(response=response, thinking=_thinking("movies_like", **kwargs))

        if name == "remember":
            content = " ".join(args).strip()
            if not content:
                raise ValueError("/remember requires text, e.g. `/remember I love sci-fi`.")
            add_long_term_memory(thread_id, content)
            return SlashCommandResult(
                response="Saved to long-term memory for this session.",
                thinking=_thinking("remember", content=content),
            )

        if name == "memory":
            memory = get_long_term_memory(thread_id)
            return SlashCommandResult(
                response=memory if memory else "(none yet)",
                thinking=_thinking("get_long_term_memory", thread_id=thread_id),
            )

        if name == "count":
            client = get_mongo_client()
            total = client.sample_mflix.movies.count_documents({})
            return SlashCommandResult(
                response=f"**{total:,}** movies in `sample_mflix.movies`.",
                thinking=_thinking("count", database="sample_mflix", collection="movies"),
            )

        if name == "genres":
            limit = _pop_value(args, "--limit", int) or 10
            client = get_mongo_client()
            pipeline = [
                {"$unwind": "$genres"},
                {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": limit},
            ]
            rows = list(client.sample_mflix.movies.aggregate(pipeline))
            if not rows:
                return SlashCommandResult(response="No genre data found.")
            lines = [f"- **{r['_id']}**: {r['count']:,}" for r in rows]
            return SlashCommandResult(
                response="**Top genres**\n" + "\n".join(lines),
                thinking=_thinking("aggregate", pipeline="top genres", limit=limit),
            )

        return SlashCommandResult(
            response="",
            error=f"Unknown command `/{name}`. Type `/help` for available commands.",
        )
    except ValueError as e:
        return SlashCommandResult(response="", error=str(e))
    except Exception as e:
        return SlashCommandResult(response="", error=f"Command failed: {e}")
