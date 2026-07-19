import asyncio

import pytest

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind, SpanStatus


def test_nested_spans_build_correct_tree(collector: TraceCollector) -> None:
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        with tokenlens.span("child-a", kind=SpanKind.LLM_CALL):
            pass
        with tokenlens.span("child-b", kind=SpanKind.TOOL):
            pass

    traces = collector.flush()
    assert len(traces) == 1

    tree = traces[0].to_tree()
    assert tree["name"] == "root"
    assert {c["name"] for c in tree["children"]} == {"child-a", "child-b"}


def test_attribution_inherits_to_children(collector: TraceCollector) -> None:
    tokenlens.set_user("user-123")
    tokenlens.set_feature("summarize")
    tokenlens.set_session("sess-1")

    with (
        tokenlens.span("root", kind=SpanKind.CHAIN),
        tokenlens.span("child", kind=SpanKind.LLM_CALL),
    ):
        pass

    traces = collector.flush()
    assert len(traces[0].spans) == 2
    for s in traces[0].spans:
        assert s.user_id == "user-123"
        assert s.feature_tag == "summarize"
        assert s.session_id == "sess-1"


def test_explicit_attribution_overrides_context(collector: TraceCollector) -> None:
    tokenlens.set_user("user-123")

    with tokenlens.span("root", kind=SpanKind.CHAIN, user_id="user-override"):
        pass

    traces = collector.flush()
    assert traces[0].root.user_id == "user-override"


async def test_concurrent_children_get_correct_parent(collector: TraceCollector) -> None:
    with tokenlens.span("parent", kind=SpanKind.CHAIN) as parent:
        parent_id = parent.span_id
        results: dict[int, str | None] = {}

        async def child(n: int) -> None:
            with tokenlens.span(f"child-{n}", kind=SpanKind.LLM_CALL) as s:
                await asyncio.sleep(0)
                results[n] = s.parent_span_id

        await asyncio.gather(child(1), child(2), child(3))

    assert results == {1: parent_id, 2: parent_id, 3: parent_id}

    traces = collector.flush()
    assert len(traces) == 1
    assert len(traces[0].spans) == 4  # parent + 3 children


def test_exception_marks_span_error_but_still_closes(collector: TraceCollector) -> None:
    with pytest.raises(RuntimeError), tokenlens.span("failing", kind=SpanKind.TOOL):
        raise RuntimeError("boom")

    traces = collector.flush()
    failed = traces[0].root
    assert failed.status == SpanStatus.ERROR
    assert failed.error_message == "boom"
    assert failed.end_time is not None


def test_child_exception_does_not_mark_sibling_or_parent_error(
    collector: TraceCollector,
) -> None:
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        with pytest.raises(RuntimeError), tokenlens.span("bad-child", kind=SpanKind.TOOL):
            raise RuntimeError("boom")
        with tokenlens.span("good-child", kind=SpanKind.TOOL):
            pass

    traces = collector.flush()
    by_name = {s.name: s for s in traces[0].spans}
    assert by_name["root"].status == SpanStatus.OK
    assert by_name["bad-child"].status == SpanStatus.ERROR
    assert by_name["good-child"].status == SpanStatus.OK


def test_child_explicit_feature_override_does_not_leak_to_siblings(
    collector: TraceCollector,
) -> None:
    tokenlens.set_feature("root-feature")

    with tokenlens.span("root", kind=SpanKind.CHAIN):
        with tokenlens.span("override-child", feature_tag="special-feature"):
            pass
        with tokenlens.span("sibling"):
            pass

    traces = collector.flush()
    by_name = {s.name: s for s in traces[0].spans}
    assert by_name["override-child"].feature_tag == "special-feature"
    assert by_name["sibling"].feature_tag == "root-feature"  # no leak
    assert by_name["root"].feature_tag == "root-feature"

    # TODO(design): calling set_feature() INSIDE a sync child span block DOES
    # leak to later siblings — span() scopes trace/span ids with contextvar
    # tokens, but attribution setters mutate the shared context directly (and
    # only asyncio task boundaries copy it). If per-span scoping is wanted,
    # span() would need to save/reset the attribution vars too. Documenting,
    # not asserting, until that call is made.


async def test_set_feature_inside_async_task_stays_in_that_task(
    collector: TraceCollector,
) -> None:
    tokenlens.set_feature("outer")

    async def child(n: int) -> None:
        if n == 1:
            tokenlens.set_feature("task-local")  # copied context: task-private
        with tokenlens.span(f"child-{n}"):
            await asyncio.sleep(0)

    with tokenlens.span("root", kind=SpanKind.CHAIN):
        await asyncio.gather(child(1), child(2))

    traces = collector.flush()
    by_name = {s.name: s for s in traces[0].spans}
    assert by_name["child-1"].feature_tag == "task-local"
    assert by_name["child-2"].feature_tag == "outer"  # unaffected sibling task
    assert by_name["root"].feature_tag == "outer"
