"""Demo RAG pipeline used for tokenlens' own tests and the README GIF.

Simulates a small retrieval-augmented pipeline with fake/canned LLMs (no
network calls, no API keys needed):

    router -> retriever -> condense_question (LLM, fails once then retries)
           -> answer_generation (LLM)

Run directly to see the resulting trace tree and token totals printed:

    python examples/demo_rag_app.py

The fake models here have no entry in the pricing table, so `cost_usd`
stays None on every span — this demo is about trace *structure* and token
attribution. See `tokenlens.cost` for the pricing/attribution engine.
"""

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

import tokenlens
from tokenlens.core.span import Trace
from tokenlens.instrument.langchain import TokenLensCallbackHandler

_DEMO_DOCS = [
    Document(page_content="tokenlens traces LLM pipelines as a tree of spans, like a profiler."),
    Document(
        page_content="Cost rolls up the span tree the same way CPU time rolls up in a flamegraph."
    ),
]


class FlakyOnceChatModel(BaseChatModel):
    """A fake chat model that fails on its first call, then succeeds.

    Stands in for a real model hitting a transient error (rate limit,
    timeout, ...) so the demo trace shows a genuine retry.
    """

    reply: str = "condensed question"
    attempts: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.attempts += 1
        if self.attempts == 1:
            raise TimeoutError("simulated transient upstream timeout")
        message = AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "flaky-once-fake"


class DemoRetriever(BaseRetriever):
    """Returns a fixed pair of documents for any query."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return _DEMO_DOCS


def build_demo_chain() -> Runnable[str, str]:
    """Build a fresh instance of the demo pipeline.

    Fresh per call so retry/attempt state never leaks between runs (e.g.
    across repeated test invocations).
    """
    router = RunnableLambda(lambda query: query.strip(), name="router")
    retriever = DemoRetriever()
    condense_model = FlakyOnceChatModel().with_retry(
        stop_after_attempt=2, wait_exponential_jitter=False
    )
    answer_model = FakeListChatModel(
        responses=["Based on the retrieved docs: tokenlens is a flamegraph for token spend."]
    )

    def run_pipeline(query: str) -> str:
        routed = router.invoke(query)
        docs = retriever.invoke(routed)
        condensed = condense_model.invoke(f"Condense this question: {routed}")
        context = "\n".join(d.page_content for d in docs)
        answer = answer_model.invoke(
            f"Context:\n{context}\n\nCondensed question: {condensed.content}"
        )
        return str(answer.content)

    return RunnableLambda(run_pipeline, name="rag_pipeline")


def run_demo(query: str = "What is tokenlens?") -> Trace:
    """Run the demo pipeline once and return its assembled Trace."""
    collector = tokenlens.get_collector()
    handler = TokenLensCallbackHandler(collector)
    chain = build_demo_chain()

    chain.invoke(query, config={"callbacks": [handler]})

    traces = collector.flush()
    assert len(traces) == 1, f"expected exactly one trace, got {len(traces)}"
    return traces[0]


def main() -> None:
    tokenlens.set_feature("rag_demo")
    trace = run_demo()

    print(json.dumps(trace.to_tree(), indent=2, default=str))
    print()
    print(f"total tokens: {trace.total_tokens()}")
    print(f"total cost:   ${trace.total_cost():.4f}  (fake models — not in the pricing table)")


if __name__ == "__main__":
    main()
