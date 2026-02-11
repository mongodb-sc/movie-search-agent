#!/usr/bin/env python3
"""
Movie recommendation AI agent: LangGraph orchestrator + Azure OpenAI + MongoDB (MCP, short-term and long-term memory).
Run: python agent.py (after setting .env with Azure OpenAI and MONGODB_URI).

Short-term memory: LangGraph checkpoint persisted to MongoDB (same cluster) via MongoDBSaver.
Long-term memory: MongoDB collection (agent_memory.long_term_memory); use the "remember" tool to store facts.
"""
import asyncio
import json
import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv()

# Lazy imports for langgraph/langchain (heavy)
def _ensure_deps():
    try:
        from langchain_openai import AzureChatOpenAI
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from langchain.agents import create_agent
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}. Install with: pip install langgraph langgraph-checkpoint-mongodb langchain-openai langchain langchain-core")
        return False

SYSTEM_PROMPT = """You are a helpful movie assistant with access to a MongoDB database (sample_mflix), a movie recommendation tool, and a semantic search over movie plots.

- For "recommend a movie", "find me a sci-fi film", "movies from the 90s" etc., use the recommend_movie tool with the appropriate genre and year range.
- For "what's that movie where...", "movies about time travel", or finding films by plot description, use the semantic_search_plots tool with a short query describing the plot or theme.
- For "movies like X", "similar to [movie title]", or "more like [movie]", use the movies_like tool with the movie title.
- For other questions about the database (e.g. "how many movies?", "top genre by count", "list collections", "explain the schema", "run a query" or aggregation), use the MongoDB MCP tools (find, aggregate, count, list-databases, list-collections, collection-schema, etc.). The movies live in database sample_mflix, collection movies. When calling MCP tools like count, find, or aggregate, always pass database="sample_mflix" and collection="movies" (unless the user asks about a different collection).
- When the user asks you to remember something (e.g. "remember I love sci-fi", "remember my favorite movie is X"), use the remember tool to store it for future conversations.
- Be concise and friendly. When recommending, mention title, year, and a short reason."""

# Custom tool names (implemented in Python). Others are MCP.
CUSTOM_TOOL_NAMES = ("recommend_movie", "semantic_search_plots", "movies_like", "remember")


def _base_mongo_uri(uri: str) -> str:
    """Return MongoDB URI without path (for checkpointer to use its own DB)."""
    parsed = urlparse(uri)
    return urlunparse((parsed.scheme, parsed.netloc, "", parsed.params, parsed.query, parsed.fragment))


def _build_custom_tools():
    """Build LangChain tools for recommend_movie, semantic_search_plots, movies_like, remember."""
    from langchain_core.tools import StructuredTool
    from tools import recommend_movie, semantic_search_plots, movies_like
    from memory import add_long_term_memory

    def _recommend(genre: str | None = None, year_min: int | None = None, year_max: int | None = None, limit: int = 5) -> str:
        return recommend_movie(genre=genre, year_min=year_min, year_max=year_max, limit=limit)

    def _semantic_search(query: str, limit: int = 5, use_reranker: bool = True) -> str:
        if not (query or "").strip():
            return "Please provide a query describing the movie or plot."
        return semantic_search_plots(query=query.strip(), limit=limit, use_reranker=use_reranker)

    def _movies_like(movie_title: str, limit: int = 5) -> str:
        return movies_like(movie_title=(movie_title or "").strip(), limit=limit)

    tools = [
        StructuredTool.from_function(
            name="recommend_movie",
            description="Recommend movies from the sample_mflix database. Use for 'recommend a movie', 'find sci-fi movies', 'movies from the 90s'. Filters by genre and/or year range.",
            func=_recommend,
        ),
        StructuredTool.from_function(
            name="semantic_search_plots",
            description="Semantic search over movie plots. Use for 'what's that movie where...', 'movies about time travel', or finding films by plot/theme description.",
            func=_semantic_search,
        ),
        StructuredTool.from_function(
            name="movies_like",
            description="Find movies similar to a given movie. Use for 'movies like [title]', 'similar to [movie]'.",
            func=_movies_like,
        ),
    ]

    import contextvars
    current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar("thread_id", default="default")

    def _remember(content: str) -> str:
        tid = current_thread_id.get()
        add_long_term_memory(tid, content)
        return "I'll remember that."

    tools.append(
        StructuredTool.from_function(
            name="remember",
            description="Store a fact for long-term memory. Use when the user asks you to remember something (e.g. 'remember I love sci-fi', 'remember my favorite movie is Inception').",
            func=_remember,
        ),
    )
    return tools, current_thread_id


def _normalize_mcp_args(args: dict) -> dict:
    """Parse JSON strings in args so MCP gets the right types (e.g. pipeline as list not string)."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and v.strip().startswith(("[", "{")):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        else:
            out[k] = v
    return out


async def _run_mcp_tool(mcp_session, name: str, args: dict) -> str:
    try:
        args = _normalize_mcp_args(args)
        out = await mcp_session.call_tool(name, args)
        if not out.content:
            return str(out)
        # Return all content parts (MCP can send e.g. summary + full result in separate parts)
        parts = []
        for c in out.content:
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else str(out)
    except Exception as e:
        return f"Tool error ({name}): {e!s}"


def _json_schema_to_pydantic(tool_name: str, props: dict, required: list[str] | None = None):
    """Build a Pydantic model from MCP JSON schema properties so the LLM gets correct tool args."""
    from pydantic import create_model, Field

    required = required or []
    fields = {}
    for k, v in (props or {}).items():
        desc = (v or {}).get("description", "")
        schema_type = (v or {}).get("type", "string")
        if schema_type == "string":
            typ = str
        elif schema_type == "integer":
            typ = int
        elif schema_type == "number":
            typ = float
        elif schema_type == "boolean":
            typ = bool
        elif schema_type == "object":
            typ = dict
        elif schema_type == "array":
            typ = list
        else:
            typ = str
        if k in required:
            fields[k] = (typ, Field(description=desc))
        else:
            fields[k] = (typ | None, Field(default=None, description=desc))
    if not fields:
        fields["_placeholder"] = (str, Field(default="", description="Unused"))
    # Unique name per tool to avoid Pydantic model name clashes
    model_name = f"MCP_{tool_name}_Args".replace("-", "_")
    return create_model(model_name, **fields)


def _build_mcp_tools_async(mcp_session, declarations_from_mcp):
    """Build async LangChain tools that call MCP from the same async task (avoids stdio cleanup errors)."""
    from langchain_core.tools import StructuredTool

    tools = []
    for decl in declarations_from_mcp:
        name = decl.get("name", "")
        if not name or name in CUSTOM_TOOL_NAMES:
            continue
        desc = decl.get("description", "")
        params = decl.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        def _make_mcp_tool(n: str, session, properties: dict):
            async def _async_mcp(**kwargs):
                # Pass only args that are in the schema; drop None to use MCP defaults
                filtered = {k: v for k, v in kwargs.items() if k in properties and v is not None}
                return await _run_mcp_tool(session, n, filtered)
            return _async_mcp

        _async_mcp = _make_mcp_tool(name, mcp_session, props)
        try:
            args_schema = _json_schema_to_pydantic(name, props, required)
        except Exception:
            args_schema = None
        tool = StructuredTool.from_function(
            name=name,
            description=desc or f"MCP tool: {name}",
            coroutine=_async_mcp,
            args_schema=args_schema,
            infer_schema=args_schema is None,
        )
        tools.append(tool)
    return tools


def _mcp_session_available() -> bool:
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        return True
    except ImportError:
        return False


async def run_agent_with_mcp() -> None:
    if not _ensure_deps():
        return

    from langchain_openai import AzureChatOpenAI
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    uri = (os.environ.get("MONGODB_URI") or "").strip()
    if not uri:
        print("MONGODB_URI not set. Set it in .env.")
        return

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    if not endpoint or not api_key:
        print("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.")
        return

    # MCP: start MongoDB MCP server and collect tool declarations
    mcp_session = None
    stdio_context = None
    mcp_declarations = []
    if _mcp_session_available():
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            mongo_uri = uri
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
                schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
                mcp_declarations.append({
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": {**schema, "required": schema.get("required", [])},
                })
            print(f"MongoDB MCP connected. {len(result.tools)} tools loaded.")
        except Exception as e:
            print(f"MCP not available: {e}. Using custom tools only.")
    else:
        if not uri:
            print("MONGODB_URI not set.")
        else:
            print("Install mcp and Node.js for MCP. Using custom tools only.")

    # Custom tools + remember
    custom_tools, thread_id_ctx = _build_custom_tools()
    all_tools = list(custom_tools)
    if mcp_session is not None and mcp_declarations:
        # Async MCP tools so graph.ainvoke runs everything on this task (clean stdio shutdown)
        mcp_tools = _build_mcp_tools_async(mcp_session, mcp_declarations)
        all_tools.extend(mcp_tools)

    # LLM
    model = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment,
        temperature=0,
        max_tokens=1024,
    )

    # Short-term memory: MongoDB checkpointer (same cluster)
    base_uri = _base_mongo_uri(uri)
    checkpoint_db = os.environ.get("LANGGRAPH_CHECKPOINT_DB", "langgraph_checkpoints")

    with MongoDBSaver.from_conn_string(base_uri, checkpoint_db) as checkpointer:
        graph = create_agent(
            model=model,
            tools=all_tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

        thread_id = os.environ.get("SESSION_ID", "default")
        config = {"configurable": {"thread_id": thread_id}}
        show_thinking = os.environ.get("SHOW_THINKING", "1").strip().lower() in ("1", "true", "yes")

        tool_names = [t.name for t in all_tools]
        print(f"Movie agent ready (LangGraph + Azure OpenAI: {deployment}).")
        print(f"Tools ({len(tool_names)}): {', '.join(sorted(tool_names))}")
        print(f"Short-term memory: MongoDB checkpointer (db={checkpoint_db}). Long-term memory: MongoDB (agent_memory.long_term_memory).")
        print(f"Session/thread_id: {thread_id}. Type 'quit' to exit.")
        if show_thinking:
            print("  (Thinking and tool calls shown. Set SHOW_THINKING=0 to hide. Use 'remember' to store facts for future sessions.)")
        else:
            print("  (Set SHOW_THINKING=1 to show tool calls and thinking.)")
        print()

        from memory import get_long_term_memory
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(None, lambda: input("You: ").strip())
            except EOFError:
                break
            if not user_input or user_input.lower() == "quit":
                break

            thread_id_ctx.set(thread_id)
            memory_text = get_long_term_memory(thread_id)
            if memory_text and memory_text != "(none yet)":
                content = f"[Remembered facts:\n{memory_text}]\n\n{user_input}"
            else:
                content = user_input
            inputs = {"messages": [HumanMessage(content=content)]}
            try:
                if show_thinking:
                    async for chunk in graph.astream(inputs, config=config, stream_mode="updates"):
                        for _node, update in chunk.items():
                            for msg in update.get("messages", []):
                                # Support both object and dict-style messages (e.g. from serialization)
                                tc = msg.get("tool_calls", []) if isinstance(msg, dict) else (getattr(msg, "tool_calls", None) or [])
                                if tc:
                                    for t in tc:
                                        # LangChain ToolCall uses top-level "name" and "args"; OpenAI uses "function": {"name", "arguments"}
                                        if isinstance(t, dict):
                                            name = t.get("name") or (t.get("function") or {}).get("name", "")
                                            raw = t.get("args") or (t.get("function") or {}).get("arguments", "{}")
                                        else:
                                            name = getattr(t, "name", "") or getattr(getattr(t, "function", None), "name", "")
                                            raw = getattr(t, "args", None) or getattr(getattr(t, "function", None), "arguments", "{}")
                                        if isinstance(raw, dict):
                                            args = json.dumps(raw)[:120]
                                        else:
                                            args = str(raw) if raw else "{}"
                                        if isinstance(args, bytes):
                                            args = args.decode("utf-8", errors="replace")
                                        args_preview = (args[:120] + "...") if len(args) > 120 else args
                                        print(f"  [thinking] → {name}({args_preview})")
                                msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
                                if msg_type == "tool":
                                    name = msg.get("name", "tool") if isinstance(msg, dict) else getattr(msg, "name", "tool")
                                    c = msg.get("content", "") if isinstance(msg, dict) else (getattr(msg, "content", "") or "")
                                    preview = (c[:200] + "...") if len(c) > 200 else c
                                    preview = preview.replace("\n", " ")
                                    print(f"  [thinking] ← {name}: {preview}")
                    state = graph.get_state(config)
                    messages = (state.values or {}).get("messages", [])
                else:
                    result = await graph.ainvoke(inputs, config=config)
                    messages = result.get("messages", [])
                content = None
                for m in reversed(messages):
                    if getattr(m, "type", "") == "ai":
                        c = getattr(m, "content", None) or ""
                        if isinstance(c, list):
                            c = " ".join(getattr(b, "text", str(b)) for b in c)
                        if (c or "").strip():
                            content = c
                            break
                print("Agent:", (content or "(no response)").strip())
            except Exception as e:
                print(f"Error: {e}")

    if mcp_session is not None:
        try:
            await mcp_session.__aexit__(None, None, None)
        except Exception:
            pass
    if stdio_context is not None:
        try:
            await stdio_context.__aexit__(None, None, None)
        except (GeneratorExit, RuntimeError, BaseExceptionGroup, Exception):
            # MCP stdio_client can raise when closing from same/different task; exit cleanly
            pass
    print("Bye.")


def main() -> None:
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("Set AZURE_OPENAI_API_KEY in .env.")
        return
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        print("Set AZURE_OPENAI_ENDPOINT in .env.")
        return
    if not os.environ.get("MONGODB_URI"):
        print("Set MONGODB_URI in .env.")
    asyncio.run(run_agent_with_mcp())


if __name__ == "__main__":
    main()
