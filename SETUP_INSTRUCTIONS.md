# Movie Agent – Setup Instructions (What You Missed)

This is the setup guide for your movie recommendation agent. Follow these steps in order.

---

## Quick answers to your original questions

| Question | Answer |
|----------|--------|
| **Does my architecture make sense?** | Yes. Azure OpenAI as the brain, MCP for MongoDB (natural-language → queries), a custom `recommend_movie` tool, and optional Voyage for embeddings/reranker is a solid design. |
| **What am I possibly missing?** | For a 30‑min demo: nothing critical. Later you could add: **Atlas Vector Search** (Voyage embeddings on plot) for “movies like X”, **Voyage reranker** after retrieval, and **structured output** (e.g. JSON) from Gemini for your app. |
| **Do I need an API key for Azure OpenAI?** | Yes (endpoint + API key + deployment name). |
| **Where do I get Azure OpenAI credentials?** | **Azure Portal** → your Azure OpenAI resource → **Keys and endpoint**; in **Azure OpenAI Studio**, create a deployment and set `AZURE_OPENAI_DEPLOYMENT`. |

---

## 30‑minute setup (do this in order)

### Step 1: Get API keys and MongoDB URI

1. **Azure OpenAI (required)**  
   - Azure Portal → your **Azure OpenAI** resource → **Keys and endpoint**.  
   - Copy **Endpoint** → save as `AZURE_OPENAI_ENDPOINT` in `.env` (e.g. `https://your-resource.openai.azure.com/`).  
   - Copy **Key** → save as `AZURE_OPENAI_API_KEY` in `.env`.  
   - In **Azure OpenAI Studio** → **Deployments** → create or use an existing deployment (e.g. gpt-4o) → set `AZURE_OPENAI_DEPLOYMENT` to that name.

2. **MongoDB Atlas (required)**  
   - In Atlas: your M10 cluster → **Connect** → **Drivers** → copy the connection string.  
   - Replace `<password>` with your database user password.  
   - Save as `MONGODB_URI` in `.env` (Step 2). If your URI has no database path (e.g. ends with `.mongodb.net/`), the agent automatically uses `sample_mflix` for MCP.

3. **Voyage AI (optional)**  
   - **https://dash.voyageai.com/** → get API key.  
   - Save as `VOYAGE_API_KEY` in `.env` if you want embeddings/reranker later.

---

### Step 2: Python 3.10+ and project setup

The **MCP package requires Python 3.10 or newer**. macOS often ships with Python 3.9, so you may need to install a newer Python first.

**Check your Python version:**

```bash
python3 --version
```

If you see **3.9.x or lower**, install Python 3.12 (or 3.10/3.11) with Homebrew:

```bash
brew install python@3.12
```

Then use that Python for the rest of the setup (e.g. `/opt/homebrew/bin/python3.12` or `$(brew --prefix python@3.12)/bin/python3.12`).

**Create `.env` and install dependencies:**

```bash
cd /Users/ajay.raghav/movie-agent

# Create .env from template
cp .env.example .env

# Edit .env and add (replace placeholders with your real values):
#   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_DEPLOYMENT=gpt-4o
#   MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
#   VOYAGE_API_KEY=pa-...   (optional)
```

**Create the virtualenv with Python 3.10+** (use the same Python you checked above):

```bash
# If you have Python 3.10+ as python3:
python3 -m venv .venv

# If you installed via Homebrew (python@3.12):
# $(brew --prefix python@3.12)/bin/python3.12 -m venv .venv

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you already created a venv with Python 3.9, remove it and recreate with 3.10+:

```bash
rm -rf .venv
$(brew --prefix python@3.12)/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Install Node.js (for MongoDB MCP server)

The MongoDB MCP server runs on Node. Your Python agent will start it automatically.

```bash
node -v
```

- If you see **v20.19+** or **v22.12+**, you’re good.  
- If not: install from **https://nodejs.org** or run:

```bash
brew install node
```

---

### Step 4: Run the agent

```bash
cd /Users/ajay.raghav/movie-agent
source .venv/bin/activate
python agent.py
```

You should see something like:  
`MongoDB MCP connected. N tools loaded...` and `Movie agent ready.`

Try in the REPL:

- *“Recommend a sci‑fi movie from the 90s.”*
- *“How many movies are in the database?”*
- *“List some action movies from 2000.”*

---

## Optional: Use MongoDB MCP inside Cursor

So that Cursor’s AI can also query your Atlas cluster (e.g. “List collections in my Atlas cluster”):

1. Create or edit **`~/.cursor/mcp.json`** (create the file if it doesn’t exist).
2. Paste this (replace `YOUR_MONGODB_URI` with your real Atlas connection string):

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

3. Restart Cursor.

You can use the project’s **cursor-mcp.json.example** as reference; it has the same structure.

---

## Summary

- **Required:** Azure OpenAI (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT), MongoDB Atlas URI, Python venv + `pip install -r requirements.txt`, Node.js for MCP.  
- **Optional:** Voyage API key (embeddings/reranker), Cursor MCP config for querying Atlas from Cursor.  
- **Sample data:** Your M10 cluster’s **sample_mflix** database (with **movies** collection) is what the agent and MCP use.

If anything fails, check: (1) `.env` has correct `AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT` and `MONGODB_URI`, (2) Node is installed and `npx` works, (3) you’re inside the venv when running `python agent.py`.  
If the agent says **"The configured connection string is not valid"**, fix `MONGODB_URI` in `.env`: use your Atlas connection string (e.g. `mongodb+srv://user:password@cluster.mongodb.net/`), replace `<password>` with your DB user password, and ensure there are no extra spaces or placeholders.

