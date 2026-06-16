"""
Shared movie agent runtime for CLI and web UI.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful movie assistant with access to a MongoDB database (sample_mflix), a movie recommendation tool, and a semantic search over movie plots.

- For "recommend a movie", "find me a sci-fi film", "movies from the 90s" etc., use the recommend_movie tool with the appropriate genre and year range.
- For "what's that movie where...", "movies about time travel", or finding films by plot description, use the semantic_search_plots tool with a short query describing the plot or theme.
- For "movies like X", "similar to [movie title]", or "more like [movie]", use the movies_like tool with the movie title.
- For other questions about the database (e.g. "how many movies?", "top genre by count", "list collections", "explain the schema", "run a query" or aggregation), use the MongoDB MCP tools (find, aggregate, count, list-databases, list-collections, collection-schema, etc.). The movies live in database sample_mflix, collection movies. When calling MCP tools like count, find, or aggregate, always pass database="sample_mflix" and collection="movies" (unless the user asks about a different collection).
- When the user asks you to remember something (e.g. "remember I love sci-fi", "remember my favorite movie is X"), use the remember tool to store it for future conversations.
- Be concise and friendly. When recommending, mention title, year, and a short reason."""

CUSTOM_TOOL_NAMES = ("recommend_movie", "semantic_search_plots", "movies_like", "remember")


@dataclass
class AgentStatus:
    ready: bool = False
    error: str | None = None
    deployment: str = ""
    tool_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    checkpoint_db: str = ""
    mcp_connected: bool = False


@dataclass
class ChatResult:
    response: str
    thinking: list[str] = field(default_factory=list)
    error: str | None = None
    direct: bool = False


def _base_mongo_uri(uri: str) -> str:
    parsed = urlparse(uri)
    return urlunparse((parsed.scheme, parsed.netloc, "", parsed.params, parsed.query, parsed.fragment))


def _ensure_deps() -> bool:
    try:
        from langchain_openai import AzureChatOpenAI  # noqa: F401
        from langgraph.checkpoint.mongodb import MongoDBSaver  # noqa: F401
        from langchain.agents import create_agent  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install with: pip install -r requirements.txt"
        ) from e


def _build_custom_tools():
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
        parts = []
        for c in out.content:
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else str(out)
    except Exception as e:
        return f"Tool error ({name}): {e!s}"


def _json_schema_to_pydantic(tool_name: str, props: dict, required: list[str] | None = None):
    from pydantic import Field, create_model

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
    model_name = f"MCP_{tool_name}_Args".replace("-", "_")
    return create_model(model_name, **fields)


def _build_mcp_tools_async(mcp_session, declarations_from_mcp):
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


def _extract_tool_call_event(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        name = tool_call.get("name") or (tool_call.get("function") or {}).get("name", "")
        raw = tool_call.get("args") or (tool_call.get("function") or {}).get("arguments", "{}")
    else:
        name = getattr(tool_call, "name", "") or getattr(getattr(tool_call, "function", None), "name", "")
        raw = getattr(tool_call, "args", None) or getattr(getattr(tool_call, "function", None), "arguments", "{}")
    if not name:
        return None
    if isinstance(raw, dict):
        args = json.dumps(raw)[:120]
    else:
        args = str(raw) if raw else "{}"
    if isinstance(args, bytes):
        args = args.decode("utf-8", errors="replace")
    args_preview = (args[:120] + "...") if len(args) > 120 else args
    return f"→ {name}({args_preview})"


def _extract_tool_result_event(msg: Any) -> str | None:
    msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
    if msg_type != "tool":
        return None
    name = msg.get("name", "tool") if isinstance(msg, dict) else getattr(msg, "name", "tool")
    content = msg.get("content", "") if isinstance(msg, dict) else (getattr(msg, "content", "") or "")
    preview = (content[:200] + "...") if len(content) > 200 else content
    preview = preview.replace("\n", " ")
    return f"← {name}: {preview}"


def _latest_ai_content(messages: list[Any]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai":
            content = getattr(m, "content", None) or ""
            if isinstance(content, list):
                content = " ".join(getattr(b, "text", str(b)) for b in content)
            if (content or "").strip():
                return content.strip()
    return "(no response)"


class MovieAgentRuntime:
    """Long-lived agent runtime shared by CLI and web UI."""

    def __init__(self) -> None:
        self._graph = None
        self._thread_id_ctx = None
        self._mcp_session = None
        self._stdio_context = None
        self._checkpointer_cm = None
        self._checkpointer = None
        self._deployment = ""
        self._tool_names: list[str] = []
        self._checkpoint_db = ""
        self._mcp_connected = False
        self._status = AgentStatus()
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def status(self) -> AgentStatus:
        return self._status

    async def initialize(self) -> AgentStatus:
        async with self._init_lock:
            if self._initialized:
                return self._status

            try:
                _ensure_deps()
                from langchain.agents import create_agent
                from langchain_openai import AzureChatOpenAI
                from langgraph.checkpoint.mongodb import MongoDBSaver

                uri = (os.environ.get("MONGODB_URI") or "").strip()
                if not uri:
                    raise ValueError("MONGODB_URI not set. Set it in .env.")

                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
                api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
                api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                if not endpoint or not api_key:
                    raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.")

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
                        self._stdio_context = stdio_client(server_params)
                        mcp_read, mcp_write = await self._stdio_context.__aenter__()
                        self._mcp_session = ClientSession(mcp_read, mcp_write)
                        await self._mcp_session.__aenter__()
                        await self._mcp_session.initialize()
                        result = await self._mcp_session.list_tools()
                        for t in result.tools:
                            schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
                            mcp_declarations.append({
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": {**schema, "required": schema.get("required", [])},
                            })
                        self._mcp_connected = True
                    except Exception:
                        self._mcp_connected = False

                custom_tools, self._thread_id_ctx = _build_custom_tools()
                all_tools = list(custom_tools)
                if self._mcp_session is not None and mcp_declarations:
                    all_tools.extend(_build_mcp_tools_async(self._mcp_session, mcp_declarations))

                model = AzureChatOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    deployment_name=deployment,
                    temperature=0,
                    max_tokens=1024,
                )

                base_uri = _base_mongo_uri(uri)
                self._checkpoint_db = os.environ.get("LANGGRAPH_CHECKPOINT_DB", "langgraph_checkpoints")
                self._checkpointer_cm = MongoDBSaver.from_conn_string(base_uri, self._checkpoint_db)
                self._checkpointer = self._checkpointer_cm.__enter__()
                self._graph = create_agent(
                    model=model,
                    tools=all_tools,
                    system_prompt=SYSTEM_PROMPT,
                    checkpointer=self._checkpointer,
                )

                self._deployment = deployment
                self._tool_names = [t.name for t in all_tools]
                self._initialized = True
                self._status = AgentStatus(
                    ready=True,
                    deployment=deployment,
                    tool_count=len(self._tool_names),
                    tool_names=sorted(self._tool_names),
                    checkpoint_db=self._checkpoint_db,
                    mcp_connected=self._mcp_connected,
                )
            except Exception as e:
                self._status = AgentStatus(ready=False, error=str(e))
                await self.shutdown()

            return self._status

    async def shutdown(self) -> None:
        if self._mcp_session is not None:
            try:
                await self._mcp_session.__aexit__(None, None, None)
            except Exception:
                pass
            self._mcp_session = None

        if self._stdio_context is not None:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except (GeneratorExit, RuntimeError, BaseExceptionGroup, Exception):
                pass
            self._stdio_context = None

        if self._checkpointer_cm is not None:
            try:
                self._checkpointer_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._checkpointer_cm = None
            self._checkpointer = None

        self._graph = None
        self._initialized = False
        if self._status.ready:
            self._status = AgentStatus()

    async def chat(
        self,
        user_input: str,
        thread_id: str | None = None,
        show_thinking: bool = True,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ChatResult:
        from slash_commands import run_slash_command

        thread_id = thread_id or os.environ.get("SESSION_ID", "default")
        slash_result = run_slash_command(user_input, thread_id=thread_id)
        if slash_result is not None:
            return ChatResult(
                response=slash_result.response,
                thinking=slash_result.thinking,
                error=slash_result.error,
                direct=True,
            )

        if not self._initialized or self._graph is None:
            status = await self.initialize()
            if not status.ready:
                return ChatResult(response="", error=status.error or "Agent failed to start.")

        from langchain_core.messages import HumanMessage
        from memory import get_long_term_memory

        self._thread_id_ctx.set(thread_id)
        config = {"configurable": {"thread_id": thread_id}}

        memory_text = get_long_term_memory(thread_id)
        if memory_text and memory_text != "(none yet)":
            content = f"[Remembered facts:\n{memory_text}]\n\n{user_input}"
        else:
            content = user_input
        inputs = {"messages": [HumanMessage(content=content)]}

        thinking: list[str] = []
        try:
            if show_thinking:
                async for chunk in self._graph.astream(inputs, config=config, stream_mode="updates"):
                    for _node, update in chunk.items():
                        for msg in update.get("messages", []):
                            tc = msg.get("tool_calls", []) if isinstance(msg, dict) else (getattr(msg, "tool_calls", None) or [])
                            for t in tc:
                                event = _extract_tool_call_event(t)
                                if event:
                                    thinking.append(event)
                                    if on_thinking:
                                        on_thinking(event)
                            result_event = _extract_tool_result_event(msg)
                            if result_event:
                                thinking.append(result_event)
                                if on_thinking:
                                    on_thinking(result_event)
                state = self._graph.get_state(config)
                messages = (state.values or {}).get("messages", [])
            else:
                result = await self._graph.ainvoke(inputs, config=config)
                messages = result.get("messages", [])

            return ChatResult(response=_latest_ai_content(messages), thinking=thinking)
        except Exception as e:
            return ChatResult(response="", thinking=thinking, error=str(e))


_runtime: MovieAgentRuntime | None = None


def get_runtime() -> MovieAgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = MovieAgentRuntime()
    return _runtime
