# Movie Agent – Setup Instructions

Step-by-step setup for the movie recommendation agent. Follow in order.

---

## What you need

| Item | Purpose |
|------|---------|
| **Azure OpenAI** | LLM (endpoint, API key, deployment name). |
| **MongoDB Atlas** | Data (`sample_mflix`), short-term memory (checkpointer), long-term memory. |
| **Python 3.10+** | Required by the `mcp` package. |
| **Node.js** | The agent starts the MongoDB MCP server via `npx`; Node is required. |
| **Voyage AI** | Semantic search and “movies like X” (embeddings + reranker). Required. |

---

## Step 1: API keys and MongoDB URI

**Azure OpenAI**

1. Azure Portal → your **Azure OpenAI** resource → **Keys and endpoint**.
2. Copy **Endpoint** → in `.env` set `AZURE_OPENAI_ENDPOINT` (e.g. `https://your-resource.openai.azure.com/`).
3. Copy **Key** → set `AZURE_OPENAI_API_KEY`.
4. In **Azure OpenAI Studio** → **Deployments** → create or use a deployment (e.g. gpt-4o) → set `AZURE_OPENAI_DEPLOYMENT` to that name.

**MongoDB Atlas**

1. Atlas → your cluster → **Connect** → **Drivers** → copy the connection string.
2. Replace `<password>` with your database user password.
3. Set `MONGODB_URI` in `.env`. The agent uses `sample_mflix` for movie data and MCP; the same cluster is used for checkpointer and long-term memory.

**Voyage AI**

- Get an API key from https://dash.voyageai.com/ and set `VOYAGE_API_KEY` in `.env`. The agent uses Voyage for embeddings and reranking (“movies like X”, plot-based search).

---

## Step 2: Project setup

**Check Python version:**

```bash
python3 --version
```

You need **3.10 or newer**. If you have 3.9 or lower, install Python 3.12 (e.g. `brew install python@3.12`) and use that for the venv.

**Create `.env` and install dependencies:**

```bash
cd movie-agent
cp .env.example .env
# Edit .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, MONGODB_URI, VOYAGE_API_KEY

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 3: Node.js (for MongoDB MCP)

The agent starts the MongoDB MCP server with `npx`. Node.js must be installed.

```bash
node -v
```

If not installed: https://nodejs.org or `brew install node`. Recommended: v20.19+ or v22.12+.

---

## Step 4: Vector search (embeddings + Atlas index)

Semantic search (“movies like X”, plot-based search) requires plot embeddings and an Atlas Vector Search index.

1. Ensure `VOYAGE_API_KEY` is set in `.env`.
2. Run once: `python embed_movies.py` (writes `plot_embedding` to each movie in `sample_mflix.movies`).
3. In Atlas → your cluster → **Database** → **sample_mflix** → **Search** → **Create Search Index**:
   - Index name: `plot_vector_index`
   - Index on: `movies` collection
   - Define: Vector Search; path `plot_embedding`; dimensions `512`; similarity cosine or dotProduct

After this, the agent can use `semantic_search_plots` and `movies_like`.

---

## Step 5: Run the agent

```bash
source .venv/bin/activate
python agent.py
```

You should see: `MongoDB MCP connected. N tools loaded.` and `Movie agent ready.`

Try:

- *“Recommend a sci-fi movie from the 90s.”*
- *“Movies like Inception.”*
- *“How many movies are in the database?”*
- *“Remember I love thrillers.”*

---

## Troubleshooting

- **Agent won’t start:** Check `.env` has `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `MONGODB_URI`, and `VOYAGE_API_KEY` with no typos or leftover placeholders.
- **“Connection string is not valid”:** Fix `MONGODB_URI`: replace `<password>`, no extra spaces, use the full Atlas connection string.
- **MCP not connecting:** Ensure Node.js is installed and `npx` works (`npx -v`). The agent spawns `npx -y mongodb-mcp-server` with your `MONGODB_URI`.
- **Python version errors:** Use Python 3.10+ for the venv. If you had a 3.9 venv, remove `.venv` and create a new one with 3.10+.
