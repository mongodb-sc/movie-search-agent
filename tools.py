"""
Movie recommendation tool and Voyage reranker.
Uses sample_mflix.movies on MongoDB Atlas.
"""
import os
from typing import Any

from pymongo import MongoClient


def get_mongo_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not set in environment")
    return MongoClient(uri)


def recommend_movie(
    genre: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 5,
) -> str:
    """
    Recommend movies from the sample_mflix database.
    Use this for user requests like "recommend a sci-fi movie" or "movies from the 90s".
    """
    client = get_mongo_client()
    db = client.sample_mflix
    coll = db.movies
    query: dict[str, Any] = {}
    if genre:
        query["genres"] = {"$regex": genre, "$options": "i"}
    if year_min is not None or year_max is not None:
        query["year"] = {}
        if year_min is not None:
            query["year"]["$gte"] = year_min
        if year_max is not None:
            query["year"]["$lte"] = year_max
    # Exclude docs with missing title/year for cleaner results
    query["title"] = {"$exists": True, "$ne": None}
    query["year"] = query.get("year") or {"$exists": True, "$type": "int"}
    cursor = coll.find(
        query,
        {"title": 1, "year": 1, "genres": 1, "plot": 1, "_id": 0},
    ).limit(limit * 2)  # fetch extra in case some filtered
    results = list(cursor)
    if not results:
        return "No movies found matching the criteria."
    out = []
    for m in results[:limit]:
        title = m.get("title") or "Unknown"
        year = m.get("year") or "?"
        genres = m.get("genres") or []
        plot = (m.get("plot") or "")[:200]
        out.append(f"- **{title}** ({year}) genres: {genres}\n  {plot}...")
    return "\n".join(out)


# Single source for Voyage embedding model and dimension (used by embed_movies.py and semantic_search_plots).
PLOT_VECTOR_INDEX_NAME = "plot_vector_index"
VOYAGE_EMBED_MODEL = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3.5-lite")
MODELS_WITH_FLEXIBLE_DIM = (
    "voyage-3", "voyage-3-large", "voyage-3.5", "voyage-3.5-lite",
    "voyage-4", "voyage-4-lite", "voyage-4-large",
    "voyage-code-3", "voyage-context-3",
)
EMBED_DIMENSION = 512 if VOYAGE_EMBED_MODEL in MODELS_WITH_FLEXIBLE_DIM else 1024


def semantic_search_plots(query: str, limit: int = 5, use_reranker: bool = True) -> str:
    """
    Semantic search over movie plots. Embeds the query with Voyage AI, runs
    Atlas Vector Search on the plot_embedding field, then reranks
    candidates with Voyage AI reranker for better relevance.

    Use for questions like "what's that movie where ..." or "movies about time travel".

    Requires: VOYAGE_API_KEY, MONGODB_URI, and an Atlas Vector Search index on
    sample_mflix.movies named plot_vector_index (path: plot_embedding; dimensions:
    512 for voyage-3.5-lite / voyage-4, or 1024 for voyage-2). Run embed_movies.py first.
    """
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return "Semantic search is not configured (VOYAGE_API_KEY missing)."
    client = get_mongo_client()
    db = client.sample_mflix
    coll = db.movies
    try:
        import voyageai
        vo = voyageai.Client(api_key=api_key)
        embed_kw = {"model": VOYAGE_EMBED_MODEL, "input_type": "query"}
        if VOYAGE_EMBED_MODEL in MODELS_WITH_FLEXIBLE_DIM:
            embed_kw["output_dimension"] = EMBED_DIMENSION
        result = vo.embed([query], **embed_kw)
        query_vector = result.embeddings[0]
    except Exception as e:
        return f"Failed to embed query: {e}"
    # Fetch more candidates if we'll rerank (reranker picks best order)
    fetch_limit = min(limit * 3, 20) if use_reranker else limit
    pipeline = [
        {
            "$vectorSearch": {
                "index": PLOT_VECTOR_INDEX_NAME,
                "path": "plot_embedding",
                "queryVector": query_vector,
                "numCandidates": min(100, fetch_limit * 20),
                "limit": fetch_limit,
            }
        },
        {"$project": {"title": 1, "year": 1, "genres": 1, "plot": 1, "_id": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        cursor = coll.aggregate(pipeline)
        results = list(cursor)
    except Exception as e:
        if "index" in str(e).lower() or "vector" in str(e).lower():
            return "Vector search index not found. Run embed_movies.py and create an Atlas Vector Search index named 'plot_vector_index' on plot_embedding (512 dimensions)."
        return f"Vector search failed: {e}"
    if not results:
        return "No matching movies found."
    if use_reranker and len(results) > limit:
        # Build doc strings for reranker (title + plot), rerank, then map back to docs
        doc_strings = []
        for m in results:
            title = m.get("title") or "Unknown"
            year = m.get("year") or "?"
            plot = (m.get("plot") or "")[:500]
            doc_strings.append(f"{title} ({year}). {plot}")
        reranked_strings = rerank_with_voyage(query=query, documents=doc_strings, top_k=limit)
        # Reorder results to match reranker order (match by doc string)
        ordered = []
        for s in reranked_strings:
            for m in results:
                t = m.get("title") or "Unknown"
                y = m.get("year") or "?"
                p = (m.get("plot") or "")[:500]
                if f"{t} ({y}). {p}" == s:
                    ordered.append(m)
                    break
        results = ordered if len(ordered) == limit else results[:limit]
    else:
        results = results[:limit]
    out = []
    for m in results:
        title = m.get("title") or "Unknown"
        year = m.get("year") or "?"
        genres = m.get("genres") or []
        plot = (m.get("plot") or "")[:200]
        score = m.get("score")
        line = f"- **{title}** ({year}) genres: {genres}\n  {plot}..."
        if score is not None:
            line += f" (score: {score:.3f})"
        out.append(line)
    return "\n".join(out)


def movies_like(movie_title: str, limit: int = 5) -> str:
    """
    Find movies similar to a given movie. Looks up the movie by title, gets its plot
    embedding from the database, runs vector search, and returns the top similar movies
    (excluding the source movie). Requires plot embeddings; run embed_movies.py first.
    """
    if not (movie_title or "").strip():
        return "Please provide a movie title."
    client = get_mongo_client()
    db = client.sample_mflix
    coll = db.movies
    # Find the movie by title (case-insensitive, partial match)
    doc = coll.find_one(
        {"title": {"$regex": f"^{movie_title.strip()}$", "$options": "i"}},
        {"plot_embedding": 1, "title": 1, "year": 1, "_id": 1},
    )
    if not doc:
        # Try contains match if exact fails
        doc = coll.find_one(
            {"title": {"$regex": movie_title.strip(), "$options": "i"}},
            {"plot_embedding": 1, "title": 1, "year": 1, "_id": 1},
        )
    if not doc:
        return f"No movie found with title matching '{movie_title.strip()}'."
    plot_embedding = doc.get("plot_embedding")
    if not plot_embedding or not isinstance(plot_embedding, list):
        return "That movie doesn't have a plot embedding. Run embed_movies.py to generate embeddings for the collection."
    source_id = doc["_id"]
    # Fetch limit+1 so we can drop the best match (the movie itself)
    pipeline = [
        {
            "$vectorSearch": {
                "index": PLOT_VECTOR_INDEX_NAME,
                "path": "plot_embedding",
                "queryVector": plot_embedding,
                "numCandidates": min(100, (limit + 1) * 20),
                "limit": limit + 1,
            }
        },
        {"$match": {"_id": {"$ne": source_id}}},
        {"$limit": limit},
        {"$project": {"title": 1, "year": 1, "genres": 1, "plot": 1, "_id": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        cursor = coll.aggregate(pipeline)
        results = list(cursor)
    except Exception as e:
        if "index" in str(e).lower() or "vector" in str(e).lower():
            return "Vector search index not found. Run embed_movies.py and create an Atlas Vector Search index named 'plot_vector_index' on plot_embedding."
        return f"Vector search failed: {e}"
    if not results:
        return "No similar movies found."
    out = [f"Movies similar to **{doc.get('title', 'Unknown')}** ({doc.get('year', '?')}):"]
    for m in results:
        title = m.get("title") or "Unknown"
        year = m.get("year") or "?"
        genres = m.get("genres") or []
        plot = (m.get("plot") or "")[:200]
        line = f"- **{title}** ({year}) genres: {genres}\n  {plot}..."
        if m.get("score") is not None:
            line += f" (score: {m['score']:.3f})"
        out.append(line)
    return "\n".join(out)


def rerank_with_voyage(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """
    Use Voyage AI reranker to reorder documents by relevance to the query.
    Requires VOYAGE_API_KEY. Returns top_k document strings in order.
    """
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return documents[:top_k]
    try:
        import voyageai
        vo = voyageai.Client(api_key=api_key)
        reranked = vo.rerank(query=query, documents=documents, model="rerank-2", top_k=top_k)
        return [documents[r.index] for r in reranked.results]
    except Exception:
        return documents[:top_k]
