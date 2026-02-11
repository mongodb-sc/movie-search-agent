# Movie Recommendation AI Agent

A movie assistant that recommends films, answers questions about the database, and remembers preferences. Uses **LangGraph** for orchestration, **Azure OpenAI** as the LLM, **MongoDB** for data and memory, and optional **Voyage AI** for semantic search.

**Quick start:** Copy `.env.example` to `.env`, set `AZURE_OPENAI_*` and `MONGODB_URI`, then `pip install -r requirements.txt` and `python agent.py`. See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for a full walkthrough.

---

## Architecture

All components and how they are used:

| Layer | Component | Role |
|-------|-----------|------|
| **Orchestrator** | LangChain `create_agent` + LangGraph | Builds and runs the ReAct agent graph (LLM → tool calls → tools → LLM until done). Compiles with a MongoDB checkpointer so conversation state is persisted per thread. |
| **LLM** | Azure OpenAI (e.g. `gpt-4o`) | Single chat model for reasoning, tool selection, and replies. Configured via `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`. |
| **Tools** | Custom (Python) + MCP | **Custom:** `recommend_movie`, `semantic_search_plots`, `movies_like`, `remember`. **MCP:** MongoDB server exposes `count`, `find`, `aggregate`, `list-collections`, etc. on `sample_mflix`. Agent picks the right tool from the user message. |
| **Short-term memory** | LangGraph + MongoDB | Conversation history per `thread_id` is stored in MongoDB via `langgraph-checkpoint-mongodb` (DB: `LANGGRAPH_CHECKPOINT_DB`, default `langgraph_checkpoints`). Same cluster as `MONGODB_URI`. |
| **Long-term memory** | MongoDB (`memory.py`) | Facts the user asks to remember are stored in `agent_memory.long_term_memory`. Loaded each turn and prepended to the user message so the LLM sees them. |
| **Embeddings** | Voyage AI (optional) | Used only for semantic search: embed query (and plot text in `embed_movies.py`) with `VOYAGE_EMBED_MODEL` (default `voyage-3.5-lite`), 512 dims. Atlas Vector Search on `plot_embedding`. |
| **Reranker** | Voyage AI (optional) | Used inside `semantic_search_plots`: Voyage `rerank-2` reorders vector-search results for relevance. |
| **Data** | MongoDB Atlas | Same cluster: `sample_mflix` (movies + vector index), checkpointer DB, and `agent_memory`. PyMongo in Python; MCP uses its own connection for its tools. |

**Flow:** User message → long-term memory loaded and prepended → graph runs with that thread’s checkpointed history → LLM may call tools (custom or MCP) → tool results appended → LLM continues until final reply.

---

## Setup

1. **Env:** `cp .env.example .env` and set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `MONGODB_URI`. Optional: `VOYAGE_API_KEY` for semantic search; `SESSION_ID`, `LANGGRAPH_CHECKPOINT_DB`, `AGENT_MEMORY_DB` for memory.
2. **Install:** `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`.
3. **Node.js:** Required for MongoDB MCP (`npx mongodb-mcp-server`). Install from [nodejs.org](https://nodejs.org) if needed.
4. **Run:** `python agent.py`. Try: “Recommend a sci-fi movie”, “How many comedy movies?”, “Remember I love thrillers.”

For vector search (“movies like X”, “what’s that movie where…”): run `python embed_movies.py` once, create an Atlas Vector Search index `plot_vector_index` on `plot_embedding` (512 dims), and set `VOYAGE_API_KEY`. See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for details.

---

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Entrypoint: LangGraph agent, tools, MCP client, memory wiring. |
| `tools.py` | `recommend_movie`, `semantic_search_plots`, `movies_like`, Voyage embed/rerank. |
| `memory.py` | Long-term memory: read/write to MongoDB `agent_memory.long_term_memory`. |
| `embed_movies.py` | One-time script to backfill `plot_embedding` with Voyage. |
| `.env.example` | Env template. |
| `EXAMPLES.md` | Sample prompts and example output. |
| `SETUP_INSTRUCTIONS.md` | **Full setup walkthrough** – use this for step-by-step setup. |
