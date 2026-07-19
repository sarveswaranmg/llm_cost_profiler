"""Pydantic response models for the API.

These define the OpenAPI schema — /docs doubles as the API documentation,
so every route returns one of these instead of a bare dict. Money fields
are Decimal and serialize as JSON strings (e.g. "0.000125") to preserve
exact micro-dollar precision; clients parse them as decimals, not floats.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tokenlens.core.span import Span


class TraceSummaryModel(BaseModel):
    """One trace, as listed in /api/traces and live-pushed on /ws/live."""

    model_config = ConfigDict(from_attributes=True)

    trace_id: str
    root_name: str
    started_at: datetime
    total_cost_usd: float
    total_tokens: int
    user_id: str | None
    feature_tag: str | None
    session_id: str | None
    has_error: bool = Field(
        default=False, description="True if any span in the trace has status ERROR."
    )


class TracePage(BaseModel):
    """A page of trace summaries, newest first."""

    traces: list[TraceSummaryModel]
    count: int = Field(description="Number of traces in this page.")
    limit: int
    offset: int


class SpanTreeNode(Span):
    """A span plus its children — the nested trace tree.

    Inherits every Span field (including the computed duration_ms /
    total_tokens); validated from Trace.to_tree() output.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=False)

    children: list["SpanTreeNode"] = Field(default_factory=list)


class FlameFrameData(BaseModel):
    """Tooltip payload attached to each flamegraph frame."""

    kind: str
    status: str
    model: str | None
    tokens: int | None
    duration_ms: float | None
    retry_index: int
    self_cost_usd: float | None


class FlameNode(BaseModel):
    """d3-flamegraph input: value is the subtree's inclusive cost in
    integer micro-dollars (a parent frame covers its children)."""

    name: str
    value: int
    data: FlameFrameData
    children: list["FlameNode"] = Field(default_factory=list)


class AggregateEntry(BaseModel):
    """One group of the cost breakdown, sorted by cost descending."""

    model_config = ConfigDict(from_attributes=True)

    key: str = Field(description="The group value, e.g. a model name or user id.")
    cost_usd: Decimal
    total_tokens: int
    call_count: int
    avg_cost_per_call: Decimal


class OverviewStats(BaseModel):
    """Headline numbers for the dashboard's top row."""

    model_config = ConfigDict(from_attributes=True)

    total_cost_usd: Decimal
    total_tokens: int
    trace_count: int
    error_rate: float = Field(
        description="Fraction of traces containing at least one ERROR span, 0..1."
    )
    retry_waste_usd: Decimal = Field(
        description="Cost of retry attempts (spans with retry_index > 0)."
    )
    top_traces: list[TraceSummaryModel] = Field(
        description="Up to 5 most expensive matching traces."
    )


class ModelPricingEntry(BaseModel):
    """Billing rates for one model, in USD per million tokens."""

    model_config = ConfigDict(from_attributes=True)

    provider: str | None
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal | None = None
    cache_write_per_mtok: Decimal | None = None


class PricingTableModel(BaseModel):
    """The active pricing table used to cost spans."""

    version: str
    source: str = Field(description='"<bundled>" or the path of a custom table.')
    models: dict[str, ModelPricingEntry]
    aliases: dict[str, str]

    @classmethod
    def from_table_dict(cls, data: dict[str, Any]) -> "PricingTableModel":
        return cls(
            version=data["version"],
            source=data["source"],
            models={
                name: ModelPricingEntry.model_validate(entry)
                for name, entry in data["models"].items()
            },
            aliases=data["aliases"],
        )
