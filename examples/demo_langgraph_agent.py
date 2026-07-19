"""Demo LangGraph agent that populates the tokenlens database.

A research agent — router → researcher → writer → critic, with the critic
able to loop back to the writer once — fully instrumented with tokenlens.
The LLMs are mocked (no network, no API keys) but use *real* model names
and realistic token counts, so the pricing engine bills every call and the
dashboard shows real dollars. Two runs include a transient rate-limit retry
and one run fails outright, so retry waste and error handling show up too.

Running it stores ~20 varied traces in the default database:

    python examples/demo_langgraph_agent.py
    tokenlens server            # → http://127.0.0.1:8321

(or just `make demo`). Set TOKENLENS_DB_PATH to write somewhere else.
"""

import random
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

import tokenlens
from tokenlens.core.span import SpanKind
from tokenlens.instrument.langgraph import instrument_graph

random.seed(20260717)

# (user, feature) pairs the fake traffic is attributed to.
_WORKLOADS = [
    ("ana@corp", "research"),
    ("ben@corp", "research"),
    ("u-1", "chat"),
    ("u-2", "summarize"),
    ("u-3", "chat"),
]

# (router model, writer model) pairs — cheap tier and premium tier.
_MODEL_TIERS = [
    ("gpt-4o-mini", "gpt-4o"),
    ("claude-haiku-4-5", "claude-sonnet-5"),
]

_QUERIES = [
    "Compare vector databases for a RAG stack under 10M documents",
    "Summarize this week's changes to the EU AI Act",
    "Why is our checkout conversion down 12% month over month?",
    "Draft a launch announcement for the new analytics API",
    "What are the tradeoffs of speculative decoding?",
    "Explain LoRA fine-tuning to a product manager",
    "Which of our features drives the most LLM spend?",
    "Write a runbook for the nightly embedding refresh job",
    "How do other teams price usage-based AI products?",
    "Investigate the latency spike in the support bot",
]

# Runs (by index) that hit a transient rate limit in the writer, and the
# one run whose researcher fails outright.
_RETRY_RUNS = {3, 11}
_ERROR_RUN = 7


class AgentState(TypedDict, total=False):
    query: str
    route: str
    notes: str
    draft: str
    verdict: str
    revisions: int
    router_model: str
    writer_model: str
    fail_researcher: bool
    retry_writer: bool


def _mock_llm_call(
    name: str,
    model: str,
    prompt: str,
    *,
    input_tokens: int,
    output_tokens: int,
    retry_index: int = 0,
    kind: SpanKind = SpanKind.LLM_CALL,
) -> str:
    """Record one (mocked) LLM call as a span; pricing bills it for real."""
    provider = "anthropic" if model.startswith("claude") else "openai"
    with tokenlens.span(
        name,
        kind=kind,
        model_name=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_preview=prompt,
        retry_index=retry_index,
    ):
        # Stand-in for network latency so durations look plausible.
        time.sleep(random.uniform(0.01, 0.05))
        return f"<{name} output, ~{output_tokens} tokens>"


def router(state: AgentState) -> AgentState:
    _mock_llm_call(
        "route_query",
        state["router_model"],
        f"Classify this request and pick a plan: {state['query']}",
        input_tokens=random.randint(150, 400),
        output_tokens=random.randint(15, 40),
    )
    return {"route": "research", "revisions": 0}


def researcher(state: AgentState) -> AgentState:
    with tokenlens.span("search_corpus", kind=SpanKind.RETRIEVER, metadata={"k": 8}):
        time.sleep(random.uniform(0.005, 0.02))
        if state.get("fail_researcher"):
            raise ConnectionError("search backend unavailable (503)")
    notes = _mock_llm_call(
        "extract_findings",
        state["router_model"],
        f"Extract key findings for: {state['query']}",
        input_tokens=random.randint(1_500, 6_000),
        output_tokens=random.randint(150, 450),
    )
    return {"notes": notes}


def writer(state: AgentState) -> AgentState:
    prompt = f"Write a grounded answer.\nNotes: {state['notes']}\nQuery: {state['query']}"
    input_tokens = random.randint(2_000, 7_000)
    if state.get("retry_writer") and state.get("revisions", 0) == 0:
        # Transient 429: record the failed attempt, then the (billed) retry.
        try:
            with tokenlens.span(
                "draft_answer",
                kind=SpanKind.LLM_CALL,
                model_name=state["writer_model"],
                provider="anthropic" if state["writer_model"].startswith("claude") else "openai",
                input_tokens=input_tokens,
                output_tokens=0,
                prompt_preview=prompt,
            ):
                raise TimeoutError("rate limited (429), retrying")
        except TimeoutError:
            pass
        draft = _mock_llm_call(
            "draft_answer",
            state["writer_model"],
            prompt,
            input_tokens=input_tokens,
            output_tokens=random.randint(300, 900),
            retry_index=1,
            kind=SpanKind.RETRY,
        )
    else:
        draft = _mock_llm_call(
            "draft_answer",
            state["writer_model"],
            prompt,
            input_tokens=input_tokens,
            output_tokens=random.randint(300, 900),
        )
    return {"draft": draft}


def critic(state: AgentState) -> AgentState:
    _mock_llm_call(
        "critique_draft",
        state["writer_model"],
        f"Critique this draft for accuracy and tone: {state['draft']}",
        input_tokens=random.randint(800, 2_500),
        output_tokens=random.randint(60, 200),
    )
    # Roughly a third of first drafts get sent back for one revision.
    needs_revision = state.get("revisions", 0) == 0 and random.random() < 0.35
    return {
        "verdict": "revise" if needs_revision else "approve",
        "revisions": state.get("revisions", 0) + 1,
    }


def _after_critic(state: AgentState) -> str:
    return "writer" if state["verdict"] == "revise" else END


def build_agent() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.set_entry_point("router")
    graph.add_edge("router", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", _after_critic)
    return instrument_graph(graph.compile())


def main() -> None:
    tokenlens.init(storage="sqlite")  # explicit, but this is also the default
    agent = build_agent()

    runs = 20
    failures = 0
    for i in range(runs):
        user, feature = _WORKLOADS[i % len(_WORKLOADS)]
        router_model, writer_model = _MODEL_TIERS[i % len(_MODEL_TIERS)]
        tokenlens.set_user(user)
        tokenlens.set_feature(feature)
        tokenlens.set_session(f"sess-{i // 4}")

        state: AgentState = {
            "query": _QUERIES[i % len(_QUERIES)],
            "router_model": router_model,
            "writer_model": writer_model,
            "retry_writer": i in _RETRY_RUNS,
            "fail_researcher": i == _ERROR_RUN,
        }
        try:
            with tokenlens.span("research_agent", kind=SpanKind.CHAIN):
                agent.invoke(state)
        except ConnectionError:
            failures += 1  # the injected hard failure — the trace still records

    traces = tokenlens.get_collector().flush()
    total_cost = sum(t.total_cost() for t in traces)
    total_tokens = sum(t.total_tokens() for t in traces)
    print(f"stored {len(traces)} traces · {total_tokens:,} tokens · ${total_cost:.4f} total")
    print(f"({failures} run(s) failed on purpose, {len(_RETRY_RUNS)} hit a simulated 429)")
    print()
    print("now explore them:")
    print("  tokenlens server        # dashboard → http://127.0.0.1:8321")
    print("  tokenlens report --group-by node")
    print("  tokenlens top")


if __name__ == "__main__":
    main()
