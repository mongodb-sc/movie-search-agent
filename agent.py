#!/usr/bin/env python3
"""
Movie recommendation AI agent: Azure OpenAI + recommend_movie + MongoDB MCP.
Run: python agent.py (after setting .env with Azure OpenAI and MONGODB_URI).
"""
import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from tools import recommend_movie, semantic_search_plots, movies_like

load_dotenv()

SYSTEM_PROMPT = """You are a helpful movie assistant with access to a MongoDB database (sample_mflix), a movie recommendation tool, and a semantic search over movie plots.

- For "recommend a movie", "find me a sci-fi film", "movies from the 90s" etc., use the recommend_movie tool with the appropriate genre and year range.
- For "what's that movie where...", "movies about time travel", or finding films by plot description, use the semantic_search_plots tool with a short query describing the plot or theme.
- For "movies like X", "similar to [movie title]", or "more like [movie]", use the movies_like tool with the movie title.
- For other questions about the database (e.g. "how many movies?", "list collections", "explain the schema", "run a query"), use the MongoDB MCP tools (find, aggregate, count, list-databases, list-collections, collection-schema, etc.). The movies live in database sample_mflix, collection movies.
- Be concise and friendly. When recommending, mention title, year, and a short reason."""

# Custom tools (implemented in Python). All other tool names are delegated to MCP.
CUSTOM_TOOL_NAMES = ("recommend_movie", "semantic_search_plots", "movies_like")

RECOMMEND_MOVIE_DECLARATION = {
    "name": "recommend_movie",
    "description": "Recommend movies from the sample_mflix database. Use for user requests like 'recommend a movie', 'find sci-fi movies', 'movies from the 90s'. Filters by genre (string) and/or year range.",
    "parameters": {
        "type": "object",
        "properties": {
            "genre": {"type": "string", "description": "Genre filter e.g. Sci-Fi, Comedy, Action"},
            "year_min": {"type": "integer", "description": "Minimum release year"},
            "year_max": {"type": "integer", "description": "Maximum release year"},
            "limit": {"type": "integer", "description": "Max number of movies to return", "default": 5},
        },
        "required": [],
    },
}

SEMANTIC_SEARCH_PLOTS_DECLARATION = {
    "name": "semantic_search_plots",
    "description": "Semantic search over movie plots. Use for questions like 'what's that movie where...', 'movies about time travel', or finding films by plot/theme description. Requires plot embeddings and Voyage API key.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language description of the plot, theme, or what happens in the movie"},
            "limit": {"type": "integer", "description": "Max number of movies to return", "default": 5},
            "use_reranker": {"type": "boolean", "description": "Use reranker for better relevance (default true)", "default": True},
        },
        "required": ["query"],
    },
}

MOVIES_LIKE_DECLARATION = {
    "name": "movies_like",
    "description": "Find movies similar to a given movie. Use when the user asks for 'movies like [title]', 'similar to [movie]', or 'more like [movie]'. Looks up the movie by title, uses its plot embedding, and returns the top 5 similar movies (excluding the same movie). Requires plot embeddings (run embed_movies.py first).",
    "parameters": {
        "type": "object",
        "properties": {
            "movie_title": {"type": "string", "description": "Title of the movie to find similar movies for (e.g. 'Iron Man', 'The Matrix')"},
            "limit": {"type": "integer", "description": "Max number of similar movies to return (default 5)", "default": 5},
        },
        "required": ["movie_title"],
    },
}


def mcp_tool_to_declaration(mcp_tool: Any) -> dict:
    """Convert MCP tool to function declaration (name, description, parameters)."""
    schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {"type": "object", "properties": {}}
    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description or "",
        "parameters": {**schema, "required": schema.get("required", [])},
    }


def run_recommend_movie(args: dict) -> str:
    genre = args.get("genre")
    year_min = args.get("year_min")
    year_max = args.get("year_max")
    limit = args.get("limit", 5)
    return recommend_movie(genre=genre, year_min=year_min, year_max=year_max, limit=limit)


def run_semantic_search_plots(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Please provide a query describing the movie or plot."
    limit = args.get("limit", 5)
    use_reranker = args.get("use_reranker", True)
    return semantic_search_plots(query=query, limit=limit, use_reranker=use_reranker)


def run_movies_like(args: dict) -> str:
    movie_title = args.get("movie_title", "").strip()
    if not movie_title:
        return "Please provide a movie title."
    limit = args.get("limit", 5)
    return movies_like(movie_title=movie_title, limit=limit)


def _mcp_session_available() -> bool:
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        return True
    except ImportError:
        return False


def _declarations_to_openai_tools(declarations: list[dict]) -> list[dict]:
    """Convert function declarations to OpenAI tools format."""
    tools = []
    for d in declarations:
        name = d.get("name", "")
        desc = d.get("description", "")
        params = d.get("parameters", {"type": "object", "properties": {}})
        tools.append({
            "type": "function",
            "function": {"name": name, "description": desc, "parameters": params},
        })
    return tools


async def _run_azure_turn(
    messages: list[dict],
    declarations: list[dict],
    mcp_session: Any,
    show_thinking: bool,
) -> str:
    """Run tool loop with Azure OpenAI. Appends assistant/tool messages to messages; returns final text."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    if not endpoint or not api_key:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    tools_openai = _declarations_to_openai_tools(declarations)
    max_tool_rounds = 10

    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools_openai,
            tool_choice="auto",
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return content or "(no response)"

        if show_thinking:
            print("  [thinking] Agent is calling tools:")
        tc_list = []
        for i, t in enumerate(tool_calls):
            fn = getattr(t, "function", None)
            tc_list.append({
                "id": getattr(t, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                },
            })
        messages.append({"role": "assistant", "content": content or "", "tool_calls": tc_list})

        for i, tc in enumerate(tool_calls):
            tid = tc_list[i]["id"]
            name = tc_list[i]["function"]["name"]
            args_s = tc_list[i]["function"]["arguments"]
            try:
                args = json.loads(args_s) if isinstance(args_s, str) else {}
            except json.JSONDecodeError:
                args = {}
            if show_thinking:
                args_preview = ", ".join(f"{k}={repr(v)}" for k, v in list(args.items())[:6])
                source = "custom" if name in CUSTOM_TOOL_NAMES else "MCP"
                print(f"    → {name}({args_preview})  [{source}]")
            if name == "recommend_movie":
                result = run_recommend_movie(args)
            elif name == "semantic_search_plots":
                result = run_semantic_search_plots(args)
            elif name == "movies_like":
                result = run_movies_like(args)
            elif mcp_session is not None:
                try:
                    out = await mcp_session.call_tool(name, args)
                    result = out.content[0].text if out.content else str(out)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = "[MCP not connected.]"
            if show_thinking:
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"    ← {name}: {preview.replace(chr(10), ' ')}")
            messages.append({"role": "tool", "tool_call_id": tid, "content": result})

    return "(max tool rounds reached)"


async def run_agent_with_mcp() -> None:
    uri = os.environ.get("MONGODB_URI")
    declarations: list[dict] = [RECOMMEND_MOVIE_DECLARATION, SEMANTIC_SEARCH_PLOTS_DECLARATION, MOVIES_LIKE_DECLARATION]
    mcp_session = None
    stdio_context = None

    if uri and _mcp_session_available():
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            mongo_uri = (uri or "").strip()
            if mongo_uri:
                base, _, qs = mongo_uri.partition("?")
                base = base.rstrip("/")
                if base.endswith(".mongodb.net"):
                    mongo_uri = base + "/sample_mflix" + ("?" + qs if qs else "")
            env = {**os.environ, "MDB_MCP_CONNECTION_STRING": mongo_uri}
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "mongodb-mcp-server", "--readOnly"],
                env=env,
            )
            stdio_context = stdio_client(server_params)
            stdio_enter = stdio_context.__aenter__()
            mcp_read, mcp_write = await stdio_enter
            mcp_session = ClientSession(mcp_read, mcp_write)
            await mcp_session.__aenter__()
            await mcp_session.initialize()
            result = await mcp_session.list_tools()
            for t in result.tools:
                declarations.append(mcp_tool_to_declaration(t))
            print(f"MongoDB MCP connected. {len(result.tools)} tools loaded (find, aggregate, count, etc.).")
        except Exception as e:
            print(f"MCP not available: {e}. Using recommend_movie only.")
    else:
        if not uri:
            print("MONGODB_URI not set. Using recommend_movie only.")
        else:
            print("Install mcp (pip install mcp) and Node.js for MCP. Using recommend_movie only.")

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    print(f"Custom tools: {', '.join(CUSTOM_TOOL_NAMES)}.")
    print(f"Movie agent ready (Azure OpenAI: {deployment}). Type 'quit' to exit.")
    if os.environ.get("SHOW_THINKING", "1").strip().lower() in ("1", "true", "yes"):
        print("  (Tool calls and results are shown. Set SHOW_THINKING=0 to hide. See EXAMPLES.md for sample prompts.)")
    print()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(None, lambda: input("You: ").strip())
            except EOFError:
                break
            if not user_input or user_input.lower() == "quit":
                break

            messages.append({"role": "user", "content": user_input})
            show_thinking = os.environ.get("SHOW_THINKING", "1").strip().lower() in ("1", "true", "yes")
            try:
                final = await _run_azure_turn(messages, declarations, mcp_session, show_thinking)
                print("Agent:", final)
            except Exception as e:
                print(f"Error: {e}")

            if len(messages) > 20:
                messages = [messages[0]] + messages[-19:]
    finally:
        if mcp_session is not None:
            try:
                await mcp_session.__aexit__(None, None, None)
            except Exception:
                pass
        if stdio_context is not None:
            try:
                await stdio_context.__aexit__(None, None, None)
            except Exception:
                pass
    print("Bye.")


def main() -> None:
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("Set AZURE_OPENAI_API_KEY in .env (Azure OpenAI API key).")
        return
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        print("Set AZURE_OPENAI_ENDPOINT in .env (e.g. https://your-resource.openai.azure.com).")
        return
    if not os.environ.get("MONGODB_URI"):
        print("Set MONGODB_URI in .env (your Atlas connection string).")
    asyncio.run(run_agent_with_mcp())


if __name__ == "__main__":
    main()
