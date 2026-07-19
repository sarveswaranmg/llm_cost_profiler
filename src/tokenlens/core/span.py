"""Span and Trace data model.

A Span represents a single traced operation (an LLM call, a retry, a graph
node, ...). A Trace is the assembled tree of Spans for one top-level
request — parent/child relationships are what let the dashboard render a
flamegraph instead of a flat table.
"""

import uuid
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

PROMPT_PREVIEW_LIMIT = 200


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SpanKind(StrEnum):
    LLM_CALL = "LLM_CALL"
    CHAIN = "CHAIN"
    GRAPH_NODE = "GRAPH_NODE"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    RETRY = "RETRY"
    CUSTOM = "CUSTOM"


class SpanStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"


class Span(BaseModel):
    """A single traced operation and, optionally, its LLM call accounting."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    span_id: str = Field(default_factory=_new_id)
    trace_id: str = Field(default_factory=_new_id)
    parent_span_id: str | None = None
    name: str
    kind: SpanKind = SpanKind.CUSTOM
    start_time: datetime = Field(default_factory=_utcnow)
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.OK
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # LLM-specific fields — nullable for non-LLM spans.
    model_name: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    prompt_preview: str | None = None
    retry_index: int = 0
    cost_usd: float | None = None

    # Attribution — inherited from context if not passed explicitly.
    user_id: str | None = None
    feature_tag: str | None = None
    session_id: str | None = None

    @field_validator("prompt_preview")
    @classmethod
    def _truncate_prompt_preview(cls, value: str | None) -> str | None:
        # Cap retained raw prompt text so a span never stores more of a
        # (possibly sensitive) prompt than a short preview needs.
        if value is None or len(value) <= PROMPT_PREVIEW_LIMIT:
            return value
        return value[:PROMPT_PREVIEW_LIMIT] + "…"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int | None:
        parts = [
            t
            for t in (
                self.input_tokens,
                self.output_tokens,
                self.cache_read_tokens,
                self.cache_write_tokens,
            )
            if t is not None
        ]
        return sum(parts) if parts else None


class Trace(BaseModel):
    """The assembled tree of spans for one top-level request."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    root: Span
    spans: list[Span]

    @classmethod
    def from_spans(cls, trace_id: str, spans: list[Span]) -> "Trace":
        roots = [s for s in spans if s.parent_span_id is None]
        if len(roots) != 1:
            raise ValueError(
                f"trace {trace_id!r} must have exactly one root span, found {len(roots)}"
            )
        return cls(trace_id=trace_id, root=roots[0], spans=spans)

    def to_tree(self) -> dict[str, Any]:
        """Return the trace as nested dicts, each with a "children" list."""
        children_by_parent: dict[str, list[Span]] = defaultdict(list)
        for s in self.spans:
            if s.parent_span_id is not None:
                children_by_parent[s.parent_span_id].append(s)

        def build(node: Span) -> dict[str, Any]:
            data = node.model_dump(mode="json")
            data["children"] = [build(child) for child in children_by_parent.get(node.span_id, [])]
            return data

        return build(self.root)

    def total_cost(self) -> float:
        return sum(s.cost_usd or 0.0 for s in self.spans)

    def total_tokens(self) -> int:
        return sum(s.total_tokens or 0 for s in self.spans)
