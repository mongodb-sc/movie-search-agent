#!/usr/bin/env python3
"""
Movie recommendation AI agent — web UI.
Run: python app.py
"""
from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from agent_service import ChatResult, get_runtime
from slash_commands import is_slash_command, run_slash_command

load_dotenv()

EXAMPLE_PROMPTS = [
    "/help",
    "/recommend sci-fi --year-min 1990 --year-max 1999",
    "/like Inception",
    "/search shark terrorizing a beach town",
    "/remember I love thrillers",
    "/memory",
    "/count",
    "/genres --limit 5",
    "Recommend a feel-good comedy",
]

CUSTOM_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: auto;
}
#header {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
#header h1 {
    font-size: 2rem;
    margin-bottom: 0.25rem;
    background: linear-gradient(90deg, #00ed64, #13aa52);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#header p {
    color: #888;
    margin: 0;
}
#status-box {
    font-size: 0.85rem;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    background: #1a1a1a;
    border: 1px solid #333;
    white-space: pre-wrap;
}
"""


def _format_status() -> str:
    runtime = get_runtime()
    status = runtime.status
    if status.ready:
        mcp = "connected" if status.mcp_connected else "unavailable"
        return (
            f"Ready — {status.deployment} | {status.tool_count} tools | MCP {mcp}\n"
            f"Checkpoint DB: {status.checkpoint_db}"
        )
    if status.error:
        return f"Not ready\n{status.error}"
    return "Starting agent…"


async def startup() -> str:
    runtime = get_runtime()
    if runtime.status.ready:
        return _format_status()
    return "UI ready — send a message to start the agent."


async def respond(
    message: str,
    history: list[dict],
    thread_id: str,
    show_thinking: bool,
):
    if not (message or "").strip():
        return history, "", _format_status()

    tid = (thread_id or "default").strip() or "default"
    text = message.strip()

    if is_slash_command(text):
        slash = run_slash_command(text, thread_id=tid)
        result = ChatResult(
            response=slash.response if slash else "",
            thinking=slash.thinking if slash else [],
            error=slash.error if slash else None,
            direct=True,
        )
    else:
        runtime = get_runtime()
        if not runtime.status.ready and not runtime.status.error:
            status = await runtime.initialize()
            if not status.ready:
                return history, "", _format_status()

        result = await runtime.chat(text, thread_id=tid, show_thinking=show_thinking)

    assistant_text = result.response
    if result.error:
        assistant_text = f"**Error:** {result.error}"
    elif show_thinking and result.thinking:
        label = "**Direct tool call**" if result.direct else "**Tool activity**"
        thinking_block = "\n".join(f"- {line}" for line in result.thinking)
        assistant_text = f"{assistant_text}\n\n---\n{label}\n{thinking_block}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_text},
    ]
    return history, "", _format_status()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Movie Agent") as demo:
        gr.HTML(
            """
            <div id="header">
              <h1>Movie Recommendation AI</h1>
              <p>Ask for recommendations, plot search, database queries, and saved preferences.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=480,
                    buttons=["copy"],
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="Message",
                        placeholder="e.g. /recommend sci-fi --year-min 1990 or ask in plain English",
                        scale=4,
                        show_label=False,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=[[p] for p in EXAMPLE_PROMPTS],
                    inputs=message,
                    label="Try an example",
                )

            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="Status",
                    value="Starting agent…",
                    interactive=False,
                    elem_id="status-box",
                    lines=4,
                )
                thread_id = gr.Textbox(
                    label="Session ID",
                    value=os.environ.get("SESSION_ID", "default"),
                    info="Same ID resumes conversation memory.",
                )
                show_thinking = gr.Checkbox(
                    label="Show tool activity",
                    value=True,
                )
                gr.Markdown(
                    """
                    **Slash commands** (no LLM): `/help`, `/recommend`, `/search`, `/like`, `/remember`, `/memory`, `/count`, `/genres`

                    **Tips**
                    - Use slash commands for faster, cheaper tool runs.
                    - Use a unique session ID per user.
                    - Plain English still works via the LLM agent.
                    """
                )

        send.click(
            respond,
            inputs=[message, chatbot, thread_id, show_thinking],
            outputs=[chatbot, message, status],
        )
        message.submit(
            respond,
            inputs=[message, chatbot, thread_id, show_thinking],
            outputs=[chatbot, message, status],
        )
        demo.load(startup, outputs=status)

    return demo


def main() -> None:
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        raise SystemExit("Set AZURE_OPENAI_API_KEY in .env.")
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT in .env.")
    if not os.environ.get("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env.")

    port = int(os.environ.get("PORT", "7860"))
    demo = build_ui()
    demo.queue().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "").strip().lower() in ("1", "true", "yes"),
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="green"),
    )


if __name__ == "__main__":
    main()
