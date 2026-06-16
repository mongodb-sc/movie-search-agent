#!/usr/bin/env python3
"""
Movie recommendation AI agent: LangGraph orchestrator + Azure OpenAI + MongoDB (MCP, short-term and long-term memory).
Run: python agent.py (after setting .env with Azure OpenAI and MONGODB_URI).

For the web UI, run: python app.py
"""
import asyncio
import os

from dotenv import load_dotenv

from agent_service import get_runtime

load_dotenv()


async def run_agent_with_mcp() -> None:
    runtime = get_runtime()
    status = await runtime.initialize()
    if not status.ready:
        print(status.error or "Agent failed to start.")
        return

    thread_id = os.environ.get("SESSION_ID", "default")
    show_thinking = os.environ.get("SHOW_THINKING", "1").strip().lower() in ("1", "true", "yes")

    print(f"Movie agent ready (LangGraph + Azure OpenAI: {status.deployment}).")
    print(f"Tools ({status.tool_count}): {', '.join(status.tool_names)}")
    print(f"Short-term memory: MongoDB checkpointer (db={status.checkpoint_db}). Long-term memory: MongoDB (agent_memory.long_term_memory).")
    print(f"Session/thread_id: {thread_id}. Type 'quit' to exit.")
    print("  Slash commands (no LLM): /help, /recommend, /search, /like, /remember, /memory, /count, /genres")
    if show_thinking:
        print("  (Thinking and tool calls shown. Set SHOW_THINKING=0 to hide. Use 'remember' to store facts for future sessions.)")
    else:
        print("  (Set SHOW_THINKING=1 to show tool calls and thinking.)")
    print()

    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(None, lambda: input("You: ").strip())
        except EOFError:
            break
        if not user_input or user_input.lower() == "quit":
            break

        result = await runtime.chat(user_input, thread_id=thread_id, show_thinking=show_thinking)
        if result.error:
            print(f"Error: {result.error}")
            continue
        if show_thinking:
            for line in result.thinking:
                print(f"  [thinking] {line}")
        print("Agent:", result.response)

    await runtime.shutdown()
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
        return
    asyncio.run(run_agent_with_mcp())


if __name__ == "__main__":
    main()
