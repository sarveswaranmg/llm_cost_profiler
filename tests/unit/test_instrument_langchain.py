import uuid

from examples.demo_rag_app import run_demo
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind, SpanStatus
from tokenlens.instrument.langchain import TokenLensCallbackHandler


def _llm_result_with_usage_metadata(
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read: int | None = None,
    cache_write: int | None = None,
) -> LLMResult:
    usage_metadata: dict[str, object] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    details = {}
    if cache_read is not None:
        details["cache_read"] = cache_read
    if cache_write is not None:
        details["cache_creation"] = cache_write
    if details:
        usage_metadata["input_token_details"] = details
    message = AIMessage(content="hi", usage_metadata=usage_metadata)  # type: ignore[arg-type]
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _llm_result_with_llm_output(prompt_tokens: int, completion_tokens: int) -> LLMResult:
    return LLMResult(
        generations=[[Generation(text="hi")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        },
    )


def test_usage_metadata_shape_captures_tokens_model_and_cache(collector: TraceCollector) -> None:
    handler = TokenLensCallbackHandler(collector)
    root_id, llm_id = uuid.uuid4(), uuid.uuid4()

    handler.on_chain_start({"name": "root"}, {}, run_id=root_id, parent_run_id=None)
    handler.on_chat_model_start(
        {"name": "chat"},
        [[]],
        run_id=llm_id,
        parent_run_id=root_id,
        metadata={"ls_model_name": "claude-x", "ls_provider": "anthropic"},
    )
    handler.on_llm_end(
        _llm_result_with_usage_metadata(100, 20, cache_read=30, cache_write=5),
        run_id=llm_id,
        parent_run_id=root_id,
    )
    handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.input_tokens == 100
    assert llm_span.output_tokens == 20
    assert llm_span.cache_read_tokens == 30
    assert llm_span.cache_write_tokens == 5
    assert llm_span.model_name == "claude-x"
    assert llm_span.provider == "anthropic"


def test_llm_output_token_usage_shape_captures_tokens(collector: TraceCollector) -> None:
    handler = TokenLensCallbackHandler(collector)
    root_id, llm_id = uuid.uuid4(), uuid.uuid4()

    handler.on_chain_start({"name": "root"}, {}, run_id=root_id, parent_run_id=None)
    handler.on_llm_start(
        {"id": ["x", "y", "SomeOpenAILLM"]}, ["hello"], run_id=llm_id, parent_run_id=root_id
    )
    handler.on_llm_end(_llm_result_with_llm_output(50, 10), run_id=llm_id, parent_run_id=root_id)
    handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.input_tokens == 50
    assert llm_span.output_tokens == 10
    assert llm_span.cache_read_tokens is None
    assert llm_span.cache_write_tokens is None


def test_retry_index_increments_under_same_parent(collector: TraceCollector) -> None:
    handler = TokenLensCallbackHandler(collector)
    root_id, attempt1, attempt2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    handler.on_chain_start({"name": "retry_wrapper"}, {}, run_id=root_id, parent_run_id=None)
    handler.on_llm_start(
        {"id": ["x", "FakeLLM"]}, ["hello"], run_id=attempt1, parent_run_id=root_id
    )
    handler.on_llm_error(RuntimeError("boom"), run_id=attempt1, parent_run_id=root_id)
    handler.on_llm_start(
        {"id": ["x", "FakeLLM"]}, ["hello"], run_id=attempt2, parent_run_id=root_id
    )
    handler.on_llm_end(_llm_result_with_llm_output(10, 5), run_id=attempt2, parent_run_id=root_id)
    handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

    traces = collector.flush()
    by_id = {s.span_id: s for s in traces[0].spans}
    assert by_id[str(attempt1)].retry_index == 0
    assert by_id[str(attempt1)].status == SpanStatus.ERROR
    assert by_id[str(attempt2)].retry_index == 1
    assert by_id[str(attempt2)].status == SpanStatus.OK


def test_sibling_llm_calls_under_different_parents_both_start_at_zero(
    collector: TraceCollector,
) -> None:
    handler = TokenLensCallbackHandler(collector)
    root_id, parent_a, parent_b, llm_a, llm_b = (uuid.uuid4() for _ in range(5))

    handler.on_chain_start({"name": "root"}, {}, run_id=root_id, parent_run_id=None)
    handler.on_chain_start({"name": "branch_a"}, {}, run_id=parent_a, parent_run_id=root_id)
    handler.on_llm_start({"id": ["FakeLLM"]}, ["a"], run_id=llm_a, parent_run_id=parent_a)
    handler.on_llm_end(_llm_result_with_llm_output(1, 1), run_id=llm_a, parent_run_id=parent_a)
    handler.on_chain_end({}, run_id=parent_a, parent_run_id=root_id)
    handler.on_chain_start({"name": "branch_b"}, {}, run_id=parent_b, parent_run_id=root_id)
    handler.on_llm_start({"id": ["FakeLLM"]}, ["b"], run_id=llm_b, parent_run_id=parent_b)
    handler.on_llm_end(_llm_result_with_llm_output(1, 1), run_id=llm_b, parent_run_id=parent_b)
    handler.on_chain_end({}, run_id=parent_b, parent_run_id=root_id)
    handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

    traces = collector.flush()
    by_id = {s.span_id: s for s in traces[0].spans}
    assert by_id[str(llm_a)].retry_index == 0
    assert by_id[str(llm_b)].retry_index == 0


def test_attribution_propagates_through_run_id_based_spans(collector: TraceCollector) -> None:
    handler = TokenLensCallbackHandler(collector)
    tokenlens.set_user("user-42")
    tokenlens.set_feature("demo")
    root_id, llm_id = uuid.uuid4(), uuid.uuid4()

    handler.on_chain_start({"name": "root"}, {}, run_id=root_id, parent_run_id=None)
    handler.on_llm_start({"id": ["FakeLLM"]}, ["hi"], run_id=llm_id, parent_run_id=root_id)
    handler.on_llm_end(_llm_result_with_llm_output(1, 1), run_id=llm_id, parent_run_id=root_id)
    handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

    traces = collector.flush()
    for s in traces[0].spans:
        assert s.user_id == "user-42"
        assert s.feature_tag == "demo"


def test_error_marks_chain_span_error_and_still_closes(collector: TraceCollector) -> None:
    handler = TokenLensCallbackHandler(collector)
    root_id = uuid.uuid4()

    handler.on_chain_start({"name": "root"}, {}, run_id=root_id, parent_run_id=None)
    handler.on_chain_error(ValueError("kaboom"), run_id=root_id, parent_run_id=None)

    traces = collector.flush()
    root = traces[0].root
    assert root.status == SpanStatus.ERROR
    assert root.error_message == "kaboom"
    assert root.end_time is not None


def test_demo_rag_app_tree_shape_matches_execution(collector: TraceCollector) -> None:
    trace = run_demo("What is tokenlens?")

    tree = trace.to_tree()
    assert tree["name"] == "rag_pipeline"
    child_names = [c["name"] for c in tree["children"]]
    assert child_names == ["router", "DemoRetriever", "FlakyOnceChatModel", "FakeListChatModel"]

    retry_wrapper = tree["children"][2]
    assert len(retry_wrapper["children"]) == 2
    first_attempt, second_attempt = retry_wrapper["children"]
    assert first_attempt["status"] == "ERROR"
    assert first_attempt["retry_index"] == 0
    assert second_attempt["status"] == "OK"
    assert second_attempt["retry_index"] == 1
    assert second_attempt["input_tokens"] == 12
    assert second_attempt["output_tokens"] == 4

    retriever_span = tree["children"][1]
    assert retriever_span["metadata"]["query"] == "What is tokenlens?"
