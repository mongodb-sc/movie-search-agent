# Movie Agent – Example Paths

When you run `python agent.py`, the agent can take different paths depending on your question. Below are example prompts and what happens under the hood (tools used and typical flow). With `SHOW_THINKING=1` (default), the console prints each tool call and result so you can see how the agent gets its information.

---

## How do I know when MCP is triggered?

Each tool call in the thinking output is labeled so you can see the source:

| Label in console | Meaning |
|------------------|--------|
| **`[custom]`** | The **recommend_movie** tool – runs in Python via PyMongo in `tools.py`. **MCP is not used.** |
| **`[MCP]`** | A tool from the **MongoDB MCP server** (e.g. `count`, `find`, `aggregate`, `list-collections`, `list-databases`, `collection-schema`). The request is sent to the Node MCP process, which talks to Atlas. **MCP is used.** |

**Rule of thumb:**  
- Only **`recommend_movie`** → custom tool, no MCP.  
- Any other tool name (`count`, `find`, `aggregate`, etc.) → MCP.

If MCP isn’t connected (e.g. no Node or no `MONGODB_URI`), you’ll only see `recommend_movie` calls; DB-style questions will fail or the agent will say it can’t run them.

---

## 1. Recommendation path (custom tool)

**You ask for a movie recommendation.** The agent uses the **`recommend_movie`** tool (queries `sample_mflix.movies` by genre/year).

| You say | What the agent typically does |
|--------|-------------------------------|
| *Recommend a sci-fi movie from the 90s.* | Calls `recommend_movie(genre='Sci-Fi', year_min=1990, year_max=1999)` → gets titles from DB → replies with a pick and short reason. |
| *Find me a comedy.* | Calls `recommend_movie(genre='Comedy')` → returns a few comedies with plot snippets. |
| *Something action from 2000–2010.* | Calls `recommend_movie(genre='Action', year_min=2000, year_max=2010)`. |

**Console (thinking on):**
```
You: Recommend a sci-fi movie from the 90s.
  [thinking] Agent is calling tools:
    → recommend_movie('genre': 'Sci-Fi', 'year_min': 1990, 'year_max': 1999)  [custom]
    ← recommend_movie: - **The Matrix** (1999) genres: ['Action', 'Sci-Fi']...
Agent: I'd recommend **The Matrix** (1999)...
```

---

## 2. Database question path (MCP tools)

**You ask about the database itself** (counts, lists, schema, ad‑hoc queries). The agent uses **MongoDB MCP tools** (e.g. `count`, `find`, `aggregate`, `list-collections`, `collection-schema`).

| You say | What the agent typically does |
|--------|-------------------------------|
| *How many movies are in the database?* | Calls MCP **`count`** on `sample_mflix.movies` → gets a number → replies. |
| *How many movies are there?* | Same as above: **`count`**. |
| *List some action movies from 2000.* | Calls MCP **`find`** (or **`aggregate`**) on `sample_mflix.movies` with filter (e.g. genre, year) → returns list. |
| *What collections exist?* | Calls MCP **`list-collections`** (and maybe **`list-databases`**) → replies with collection names. |
| *What’s the schema of the movies collection?* | Calls MCP **`collection-schema`** (or similar) for `sample_mflix.movies` → describes fields. |
| *Give me 5 movies from 1995.* | Calls MCP **`find`** with year filter and limit → lists titles/years. |

**Console (thinking on):**
```
You: How many movies are in the database?
  [thinking] Agent is calling tools:
    → count('database': 'sample_mflix', 'collection': 'movies', ...)  [MCP]
    ← count: 23541
Agent: There are 23,541 movies in the database.
```

---

## 3. Mixed path (tool + then answer)

**You ask something that might need one or more tool calls, then a final answer.** The agent can chain: e.g. call a tool, get data, then produce a natural-language reply (and optionally call another tool in a second round).

| You say | What might happen |
|--------|--------------------|
| *How many sci-fi movies are there?* | MCP **`count`** or **`aggregate`** with genre filter → agent replies with the number. |
| *Recommend something and also tell me how many movies are in the DB.* | May call **`recommend_movie`** and MCP **`count`** (order can vary) → one reply with both. |
| *What’s the oldest movie in the database?* | MCP **`find`** or **`aggregate`** (e.g. sort by year, limit 1) → agent states title and year. |

---

## 4. Disable thinking in the console

To hide the tool-call / result lines and only see your prompt and the agent’s final answer:

```bash
SHOW_THINKING=0 python agent.py
```

Or in `.env`:
```
SHOW_THINKING=0
```

---

## Quick reference: tools used per path

| Path | Main tools | Data source |
|------|------------|-------------|
| “Recommend a movie / find a film” | `recommend_movie` | `sample_mflix.movies` (PyMongo in `tools.py`) |
| “How many… / list… / what collections / schema” | MCP: `count`, `find`, `aggregate`, `list-collections`, `collection-schema`, etc. | MongoDB via MCP server |

The agent (Gemini) decides which tool to use from your natural language; the thinking output in the console shows exactly which tools were called and with what arguments, so you can see how it got the information.
