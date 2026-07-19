"""LangChain instrumentation.

TokenLensCallbackHandler maps LangChain's callback events onto tokenlens
spans. Parentage is built from LangChain's own run_id/parent_run_id (each
span's id IS the stringified run_id) rather than tokenlens' contextvars
stack: LangChain dispatches callbacks in whatever order its own execution
engine runs steps, which isn't guaranteed to be a well-nested call stack the
way `tokenlens.span()` expects. Keying spans by run_id sidesteps that and
guarantees the resulting tree matches LangChain's actual execution graph.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tokenlens.core.collector import TraceCollector, get_collector
from tokenlens.core.context import current_attribution
from tokenlens.core.span import PROMPT_PREVIEW_LIMIT, Span, SpanKind, SpanStatus

_ROOT_RETRY_KEY = "__root__"


def _preview(text: str | None, limit: int = PROMPT_PREVIEW_LIMIT) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "…"


def _resolve_name(
    serialized: dict[str, Any] | None, *, fallback: str, explicit_name: str | None = None
) -> str:
    if explicit_name:
        return explicit_name
    if serialized:
        name = serialized.get("name")
        if name:
            return str(name)
        id_path = serialized.get("id")
        if isinstance(id_path, list) and id_path:
            return str(id_path[-1])
    return fallback


def _resolve_model(
    serialized: dict[str, Any] | None,
    invocation_params: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    invocation_params = invocation_params or {}
    metadata = metadata or {}

    model_name = (
        invocation_params.get("model_name")
        or invocation_params.get("model")
        or metadata.get("ls_model_name")
    )
    provider = metadata.get("ls_provider") or invocation_params.get("_type")
    return (
        str(model_name) if model_name else None,
        str(provider) if provider else None,
    )


def _preview_from_lc_messages(messages: list[list[Any]]) -> str | None:
    if not messages or not messages[0]:
        return None
    content = messages[0][-1].content
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
    return str(content) if content is not None else None


def _usage_from_generations(response: LLMResult) -> dict[str, int | None] | None:
    for generation_list in response.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None)
            if not usage_metadata:
                continue
            input_details = usage_metadata.get("input_token_details") or {}
            return {
                "input_tokens": usage_metadata.get("input_tokens"),
                "output_tokens": usage_metadata.get("output_tokens"),
                "cache_read_tokens": input_details.get("cache_read"),
                "cache_write_tokens": input_details.get("cache_creation"),
            }
    return None


def _usage_from_llm_output(response: LLMResult) -> dict[str, int | None]:
    llm_output = response.llm_output or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    return {
        "input_tokens": token_usage.get("prompt_tokens"),
        "output_tokens": token_usage.get("completion_tokens"),
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }


def _extract_usage(response: LLMResult) -> dict[str, int | None]:
    # Newer LangChain: usage lives on each generation's AIMessage.usage_metadata
    # (this is the only shape that carries cache token counts). Older/plain
    # LLM integrations only ever populate llm_output["token_usage"].
    return _usage_from_generations(response) or _usage_from_llm_output(response)


class TokenLensCallbackHandler(BaseCallbackHandler):
    """Feeds tokenlens spans from LangChain callback events.

    Attach one instance per invocation (or share one across invocations —
    run ids are unique per run, so there's no cross-talk):
    `chain.invoke(x, config={"callbacks": [TokenLensCallbackHandler()]})`.
    """

    def __init__(self, collector: TraceCollector | None = None) -> None:
        self._collector = collector or get_collector()
        self._spans: dict[str, Span] = {}
        # Retry attempts are tracked per parent run: LangChain's built-in
        # retry (`.with_retry()`) re-invokes the same logical call with a
        # fresh run_id but the same parent_run_id on every attempt. A true
        # sibling fan-out of independent LLM calls under one parent would
        # also increment this counter — a known, accepted simplification.
        self._retry_counts: dict[str, int] = {}

    # -- span bookkeeping -------------------------------------------------

    def _trace_id_for(self, run_id: UUID, parent_run_id: UUID | None) -> str:
        if parent_run_id is not None:
            parent = self._spans.get(str(parent_run_id))
            if parent is not None:
                return parent.trace_id
        # No known parent: this run is itself the root of a new trace, so it
        # becomes its own trace id.
        return str(run_id)

    def _open_span(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        kind: SpanKind,
        **span_kwargs: Any,
    ) -> Span:
        parent_span_id = str(parent_run_id) if parent_run_id is not None else None
        for key, value in current_attribution().items():
            span_kwargs.setdefault(key, value)
        current = Span(
            span_id=str(run_id),
            trace_id=self._trace_id_for(run_id, parent_run_id),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            **span_kwargs,
        )
        self._spans[str(run_id)] = current
        return current

    def _close_span(
        self,
        run_id: UUID,
        *,
        status: SpanStatus = SpanStatus.OK,
        error_message: str | None = None,
    ) -> None:
        current = self._spans.pop(str(run_id), None)
        if current is None:
            return
        current.status = status
        current.error_message = error_message
        current.end_time = datetime.now(UTC)
        # No further children can start under a run_id whose own *_end/*_error
        # callback has already fired, so it's safe to forget its retry count.
        self._retry_counts.pop(str(run_id), None)
        self._collector.record(current)

    # -- chains -------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _resolve_name(serialized, fallback="chain", explicit_name=kwargs.get("name"))
        self._open_span(run_id=run_id, parent_run_id=parent_run_id, name=name, kind=SpanKind.CHAIN)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, status=SpanStatus.ERROR, error_message=str(error))

    # -- LLMs -----------------------------------------------------------------

    def _start_llm_span(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        serialized: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        invocation_params: dict[str, Any] | None,
        prompt_preview: str | None,
    ) -> None:
        model_name, provider = _resolve_model(serialized, invocation_params, metadata)
        retry_key = str(parent_run_id) if parent_run_id is not None else _ROOT_RETRY_KEY
        retry_index = self._retry_counts.get(retry_key, 0)
        self._retry_counts[retry_key] = retry_index + 1

        self._open_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=model_name or _resolve_name(serialized, fallback="llm"),
            kind=SpanKind.LLM_CALL,
            model_name=model_name,
            provider=provider,
            prompt_preview=prompt_preview,
            retry_index=retry_index,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_llm_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            metadata=metadata,
            invocation_params=invocation_params or kwargs.get("invocation_params"),
            prompt_preview=_preview(prompts[0] if prompts else None),
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_llm_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            metadata=metadata,
            invocation_params=invocation_params or kwargs.get("invocation_params"),
            prompt_preview=_preview(_preview_from_lc_messages(messages)),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        current = self._spans.get(str(run_id))
        if current is not None:
            usage = _extract_usage(response)
            current.input_tokens = usage["input_tokens"]
            current.output_tokens = usage["output_tokens"]
            current.cache_read_tokens = usage["cache_read_tokens"]
            current.cache_write_tokens = usage["cache_write_tokens"]
        self._close_span(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, status=SpanStatus.ERROR, error_message=str(error))

    # -- retrievers -----------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _resolve_name(serialized, fallback="retriever", explicit_name=kwargs.get("name"))
        self._open_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            kind=SpanKind.RETRIEVER,
            metadata={"query": _preview(query)} if query else {},
        )

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, status=SpanStatus.ERROR, error_message=str(error))

    # -- tools ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _resolve_name(serialized, fallback="tool", explicit_name=kwargs.get("name"))
        self._open_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            kind=SpanKind.TOOL,
            metadata={"input": _preview(input_str)} if input_str else {},
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._close_span(run_id, status=SpanStatus.ERROR, error_message=str(error))
