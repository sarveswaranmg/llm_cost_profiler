"""Instrumentation adapters.

Adapters that convert framework-specific callbacks/hooks (LangChain,
LangGraph, raw OpenAI/Anthropic SDKs) into tokenlens spans.

`auto_patch()` is the zero-config entry point for the raw SDKs: it
monkey-patches whichever of openai/anthropic are installed so their
chat/messages `create()` calls become traced LLM_CALL spans. LangChain and
LangGraph integration is opt-in via `tokenlens.instrument.langchain` and
`tokenlens.instrument.langgraph` since those pull in extra dependencies.
"""


def auto_patch() -> None:
    """Patch installed LLM SDKs (OpenAI, Anthropic) for automatic tracing.

    Silently skips whichever SDK isn't installed. Safe to call more than
    once — each SDK's client method is only wrapped the first time.
    """
    from tokenlens.instrument import anthropic_sdk, openai_sdk

    openai_sdk.patch()
    anthropic_sdk.patch()


__all__ = ["auto_patch"]
