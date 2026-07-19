"""Small helpers shared by the raw-SDK instrumentation adapters."""

from typing import Any


def preview_from_messages(messages: Any) -> str | None:
    """Best-effort text preview of the last message in a chat-style message list.

    Handles both the plain string `content` shape and the multimodal
    list-of-parts shape used by the OpenAI/Anthropic chat APIs.
    """
    if not messages:
        return None
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        text = " ".join(p for p in parts if p)
        return text or None
    return None
