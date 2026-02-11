#!/usr/bin/env python3
"""
Add plot embeddings to all documents in sample_mflix.movies using Voyage AI.

Run once to populate the `plot_embedding` field. Then create an Atlas Vector Search
index on that field and use the semantic_search_plots tool in the agent.

Usage:
  python embed_movies.py                    # embed all docs missing plot_embedding
  python embed_movies.py --limit 100       # only first 100 (for testing)
  python embed_movies.py --overwrite       # re-embed docs that already have plot_embedding
  python embed_movies.py --dry-run         # print counts only, no writes

Requires: MONGODB_URI, VOYAGE_API_KEY in .env.
Uses the same model and dimension as tools.semantic_search_plots (defined in tools.py;
set VOYAGE_EMBED_MODEL in .env to change).

After running, create an Atlas Vector Search index on sample_mflix.movies:
  - Index name: plot_vector_index
  - Field: plot_embedding (type: vector)
  - Dimensions: 512 or 1024 (must match EMBED_DIMENSION used here)
  - Similarity: cosine (or dotProduct)
Then add the semantic_search_plots tool to the agent for "what's that movie where..." queries.
"""
import argparse
import os
import time
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from tools import (
    EMBED_DIMENSION,
    MODELS_WITH_FLEXIBLE_DIM,
    VOYAGE_EMBED_MODEL,
    get_mongo_client,
)

load_dotenv()

BATCH_SIZE = 128  # Voyage API batch limit
EMBEDDING_FIELD = "plot_embedding"


def _get_mongo_client() -> MongoClient:
    try:
        return get_mongo_client()
    except ValueError as e:
        raise SystemExit(str(e)) from e


def get_voyage_client():
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("VOYAGE_API_KEY not set. Add it to .env")
    import voyageai
    return voyageai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Voyage AI plot embeddings to sample_mflix.movies")
    parser.add_argument("--limit", type=int, default=0, help="Max number of documents to process (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Re-embed documents that already have plot_embedding")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, do not call Voyage or write to DB")
    args = parser.parse_args()

    client = _get_mongo_client()
    db = client.sample_mflix
    coll = db.movies

    # Build query: must have non-empty plot; optionally skip already-embedded
    query: dict[str, Any] = {
        "plot": {"$exists": True, "$ne": None, "$type": "string"},
        "$expr": {"$gt": [{"$strLenCP": "$plot"}, 0]},
    }
    if not args.overwrite:
        query[EMBEDDING_FIELD] = {"$exists": False}

    total = coll.count_documents(query)
    if total == 0:
        print("No documents to process.")
        return
    if args.limit:
        total = min(total, args.limit)
    print(f"Documents to process: {total} (overwrite={args.overwrite})")

    if args.dry_run:
        print("Dry run: exiting without calling Voyage or updating DB.")
        return

    vo = get_voyage_client()
    cursor = coll.find(query, {"_id": 1, "plot": 1}).limit(args.limit or 0)
    processed = 0
    batch_ids: list[Any] = []
    batch_texts: list[str] = []

    def flush_batch() -> None:
        nonlocal processed
        if not batch_ids:
            return
        embed_kw: dict = {"model": VOYAGE_EMBED_MODEL, "input_type": "document"}
        if VOYAGE_EMBED_MODEL in MODELS_WITH_FLEXIBLE_DIM:
            embed_kw["output_dimension"] = EMBED_DIMENSION
        try:
            result = vo.embed(batch_texts, **embed_kw)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                print("Rate limited; waiting 60s...")
                time.sleep(60)
                result = vo.embed(batch_texts, **embed_kw)
            else:
                raise
        embeddings = result.embeddings
        ops = [
            UpdateOne(
                {"_id": bid},
                {"$set": {EMBEDDING_FIELD: emb}},
            )
            for bid, emb in zip(batch_ids, embeddings)
        ]
        coll.bulk_write(ops)
        processed += len(batch_ids)
        print(f"  Embedded {processed}/{total} documents")

    for doc in cursor:
        plot = (doc.get("plot") or "").strip()
        if not plot:
            continue
        batch_ids.append(doc["_id"])
        batch_texts.append(plot)
        if len(batch_ids) >= BATCH_SIZE:
            flush_batch()
            batch_ids.clear()
            batch_texts.clear()

    flush_batch()
    print(f"Done. Total documents updated: {processed}.")


if __name__ == "__main__":
    main()
