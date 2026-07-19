# Integrations

All integrations produce the same thing — a tree of `Span`s per request —
so they compose freely: a LangGraph node that internally calls LangChain
which calls the OpenAI SDK nests into one coherent trace.

## LangChain

```python
from tokenlens.instrument.langchain import TokenLensCallbackHandler

handler = TokenLensCallbackHandler()
chain.invoke(query, config={"callbacks": [handler]})
```

The handler mirrors LangChain's run tree: chains become `CHAIN` spans, LLM
calls become `LLM_CALL` spans with token usage from `usage_metadata`,
retrievers become `RETRIEVER` spans, tools become `TOOL` spans. Retried
model calls carry an incremented `retry_index`.

## LangGraph

```python
import tokenlens
from tokenlens.instrument.langgraph import instrument_graph

agent = instrument_graph(builder.compile())

with tokenlens.span("my_agent"):        # ties all node spans into one trace
    agent.invoke(state)
```

`instrument_graph` wraps every node in a `GRAPH_NODE` span, sync or async,
including parallel fan-out. Wrap the top-level `invoke` in a span of your
own (as above) so the nodes share one trace; otherwise each node run becomes
its own trace. For hand-rolled graphs there's a decorator:

```python
from tokenlens.instrument.langgraph import traced_node

@traced_node("researcher")
def researcher(state): ...
```

## Raw OpenAI / Anthropic SDKs

```python
import tokenlens.instrument

tokenlens.instrument.auto_patch()
```

Patches `openai` chat completions and `anthropic` messages create calls
(sync and async) to record `LLM_CALL` spans with model, token usage
(including cache read/write tokens), and a truncated prompt preview.
Only installed SDKs are patched; calling it twice is a no-op.

## Manual spans

The primitives underneath are public and cheap:

```python
import tokenlens
from tokenlens import SpanKind

with tokenlens.span("rerank_results", kind=SpanKind.TOOL, metadata={"k": 8}):
    ...

with tokenlens.span(
    "call_llm",
    kind=SpanKind.LLM_CALL,
    model_name="gpt-4o",           # priced automatically
    input_tokens=1200,
    output_tokens=300,
):
    ...
```

Spans nest by `contextvars`, so nesting is correct across `await`s, asyncio
task fan-out, and threads (each thread/task inherits its creation context).
Exceptions mark the span `ERROR` (with the message) and re-raise.

## Attribution

```python
tokenlens.set_user("ana@corp")
tokenlens.set_feature("support_bot")
tokenlens.set_session("sess-42")
```

Set once per request; every descendant span inherits the values, and every
view (dashboard, `tokenlens report`, `/api/aggregate`) can group by them.
