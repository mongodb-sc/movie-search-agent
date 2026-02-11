# Movie Recommendation AI Agent

Azure OpenAI–powered agent with MongoDB (MCP), movie recommendation tool, and optional Voyage AI embeddings/reranker. Uses the **sample_mflix** database (movies collection) on your Atlas M10 cluster.

> **→ Step-by-step setup:** see **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** for the full 30‑minute setup and answers to common questions (API keys, architecture, what you might add later).

---

## Architecture (and what you have)

| Component | Role |
|-----------|------|
| **Azure OpenAI** | Agent “brain”: decides which tool to use, turns natural language into queries, answers the user. |
| **MongoDB MCP server** | Exposes DB as tools (find, aggregate, list collections, etc.) so the agent can run natural-language-driven queries. |
| **Recommend-movie tool** | Custom tool for “recommend a movie” style requests (genre, year, etc.). |
| **Voyage AI** | Optional: embeddings for semantic search, reranker to improve result order. |

**Does it make sense?** Yes. You get:
- **Recommendation flow**: User asks for a movie → agent uses `recommend_movie` (and/or MCP find/aggregate on `sample_mflix.movies`).
- **General DB flow**: User asks ad‑hoc questions → Azure OpenAI chooses MCP tools and builds queries.

**What you might add later (not required for 30‑min setup):**
- **Atlas Vector Search** on a movies collection (e.g. plot embeddings with Voyage) for “movies like X” semantic search.
- **Reranker**: After retrieval (keyword or vector), pass candidates to Voyage reranker for better ordering.
- **Structured output** from the model (e.g. JSON) so your app can parse recommendations reliably.

---

## 30‑minute setup

### 1. API keys and config

- **Azure OpenAI (required)**  
  - Azure Portal → your Azure OpenAI resource → **Keys and endpoint**.  
  - Put **Endpoint** in `.env` as `AZURE_OPENAI_ENDPOINT` (e.g. `https://your-resource.openai.azure.com/`).  
  - Put **Key** in `.env` as `AZURE_OPENAI_API_KEY`.  
  - In Azure OpenAI Studio, create a deployment (e.g. gpt-4o) and set `AZURE_OPENAI_DEPLOYMENT` to that name.

- **MongoDB Atlas**  
  - Atlas → your cluster → **Connect** → **Drivers** → copy the connection string. Replace `<password>`, put in `.env` as `MONGODB_URI`.

- **Voyage AI (optional)**  
  - **https://dash.voyageai.com/** → API key → `.env` as `VOYAGE_API_KEY`.

### 2. Project setup

```bash
cd movie-agent
cp .env.example .env
# Edit .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, MONGODB_URI

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Node.js (for MongoDB MCP)

The MongoDB MCP server is Node-based. Your Python agent starts it as a subprocess.

```bash
node -v   # Need 20.19+ or 22.12+
# If missing: install from https://nodejs.org or: brew install node
```

### 4. Run the agent

```bash
source .venv/bin/activate
python agent.py
```

Then try:
- “Recommend a sci‑fi movie from the 90s.”
- “How many movies are in the database?”
- “List some action movies from 2000.”

---

## Cursor IDE: MongoDB MCP (optional)

To use MongoDB MCP from Cursor (so Cursor’s AI can query your cluster):

1. Create or edit **`~/.cursor/mcp.json`** (create the file if it doesn’t exist).
2. Add the MongoDB server (replace `YOUR_MONGODB_URI` with your real URI):

```json
{
  "mcpServers": {
    "mongodb": {
      "command": "npx",
      "args": ["-y", "mongodb-mcp-server", "--readOnly"],
      "env": {
        "MDB_MCP_CONNECTION_STRING": "YOUR_MONGODB_URI"
      }
    }
  }
}
```

3. Restart Cursor. In chat you can ask things like “List collections in my Atlas cluster” or “Find movies from 1999.”

---

## Files

- `agent.py` – Main agent: Gemini + MCP client (spawns MongoDB MCP) + `recommend_movie` tool. Tool calls/results are printed when `SHOW_THINKING` is on (default).
- `tools.py` – Movie recommendation tool, `semantic_search_plots` (vector search on plot), and optional Voyage reranker.
- `embed_movies.py` – One-time script to add Voyage AI plot embeddings (512 dims) to all movies for vector search.
- `EXAMPLES.md` – Example prompts and paths (recommendation vs DB questions, with sample console output).
- `OLLAMA_SETUP.md` – Install Ollama and a tool-capable model for local fallback when Gemini is rate limited.
- `.env.example` – Template for API keys and `MONGODB_URI`.
- `cursor-mcp.json.example` – Example for `~/.cursor/mcp.json`.

---

## Vector search (semantic “what’s that movie where…”)

1. **Add embeddings** (once):  
   `python embed_movies.py`  
   Requires `VOYAGE_API_KEY`. Default model **voyage-3.5-lite** (512 dims); set `VOYAGE_EMBED_MODEL` in `.env` to any embedding model from your account (e.g. voyage-4-lite). Writes `plot_embedding` to each document from `plot`.

2. **Create an Atlas Vector Search index** on `sample_mflix.movies`:  
   - Index name: **`plot_vector_index`**  
   - Path: **`plot_embedding`**  
   - Dimensions: **512** (default with voyage-3.5-lite) or 1024 if using an older fixed-dim model.  
   - Similarity: cosine (or dotProduct).

3. **Use in the agent**: The helper `tools.semantic_search_plots(query, limit=5)` embeds the query with Voyage and runs `$vectorSearch`. You can expose it as an agent tool for questions like “what’s that movie where they go back in time?”.

---

## Sample data

Atlas “Load sample dataset” includes **sample_mflix** with a **movies** collection (title, year, genres, etc.). The agent and MCP are configured to use this database.
