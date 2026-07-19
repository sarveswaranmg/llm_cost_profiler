"""API routes.

REST endpoints for traces, aggregation, stats, and pricing, plus the
/ws/live WebSocket that streams trace summaries as the collector flushes
them. All handlers are thin: query parsing here, math in tokenlens.cost,
persistence in tokenlens.storage.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket
from starlette.websockets import WebSocketDisconnect

from tokenlens.cost.attribution import trace_flame_data
from tokenlens.cost.pricing import get_default_table
from tokenlens.server.models import (
    AggregateEntry,
    FlameNode,
    OverviewStats,
    PricingTableModel,
    SpanTreeNode,
    TracePage,
    TraceSummaryModel,
)
from tokenlens.storage.base import StorageBackend, TraceFilters

api = APIRouter(prefix="/api")
ws = APIRouter()


class GroupBy(StrEnum):
    """Dimension to break down cost by."""

    user_id = "user_id"
    feature_tag = "feature_tag"
    model = "model"
    node = "node"
    kind = "kind"


# API-facing names → storage/attribution grouping keys.
_GROUP_BY_KEY = {
    GroupBy.user_id: "user_id",
    GroupBy.feature_tag: "feature_tag",
    GroupBy.model: "model_name",
    GroupBy.node: "node_name",
    GroupBy.kind: "kind",
}


def _backend(request: Request) -> StorageBackend:
    backend: StorageBackend = request.app.state.backend
    return backend


def _filters(
    user_id: Annotated[
        str | None, Query(description="Only traces attributed to this user.")
    ] = None,
    feature_tag: Annotated[
        str | None, Query(description="Only traces with this feature tag.")
    ] = None,
    model: Annotated[
        str | None, Query(description="Only traces containing a span that called this model.")
    ] = None,
    since: Annotated[
        datetime | None, Query(description="Only traces started at/after this time (naive = UTC).")
    ] = None,
    until: Annotated[
        datetime | None, Query(description="Only traces started at/before this time (naive = UTC).")
    ] = None,
    min_cost: Annotated[
        float | None, Query(ge=0, description="Only traces costing at least this many USD.")
    ] = None,
) -> TraceFilters:
    return TraceFilters(
        user_id=user_id,
        feature_tag=feature_tag,
        model=model,
        since=since,
        until=until,
        min_cost=min_cost,
    )


Backend = Annotated[StorageBackend, Depends(_backend)]
Filters = Annotated[TraceFilters, Depends(_filters)]


@api.get(
    "/traces",
    response_model=TracePage,
    tags=["traces"],
    summary="List traces",
    description="Paginated trace summaries, newest first. All filters combine with AND.",
)
async def list_traces(
    backend: Backend,
    filters: Filters,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TracePage:
    summaries = await backend.list_traces(limit=limit, offset=offset, filters=filters)
    traces = [TraceSummaryModel.model_validate(s) for s in summaries]
    return TracePage(traces=traces, count=len(traces), limit=limit, offset=offset)


@api.get(
    "/traces/{trace_id}",
    response_model=SpanTreeNode,
    tags=["traces"],
    summary="Get one trace as a span tree",
    responses={404: {"description": "Unknown trace id."}},
)
async def get_trace(trace_id: str, backend: Backend) -> SpanTreeNode:
    trace = await backend.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    return SpanTreeNode.model_validate(trace.to_tree())


@api.get(
    "/traces/{trace_id}/flame",
    response_model=FlameNode,
    tags=["traces"],
    summary="Get one trace in d3-flamegraph format",
    responses={404: {"description": "Unknown trace id."}},
)
async def get_trace_flame(trace_id: str, backend: Backend) -> FlameNode:
    trace = await backend.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    return FlameNode.model_validate(trace_flame_data(trace))


@api.get(
    "/aggregate",
    response_model=list[AggregateEntry],
    tags=["aggregate"],
    summary="Cost breakdown by dimension",
    description="Cost/token/call totals grouped by the chosen dimension, most expensive "
    "first, over every span of the traces matching the filters.",
)
async def aggregate(group_by: GroupBy, backend: Backend, filters: Filters) -> list[AggregateEntry]:
    entries = await backend.aggregate(_GROUP_BY_KEY[group_by], filters=filters)
    return [AggregateEntry.model_validate(e) for e in entries]


@api.get(
    "/stats/overview",
    response_model=OverviewStats,
    tags=["stats"],
    summary="Headline stats",
    description="Totals for the dashboard: cost, tokens, trace count, error rate, "
    "retry waste, and the 5 most expensive traces.",
)
async def stats_overview(
    backend: Backend,
    since: Annotated[
        datetime | None, Query(description="Only traces started at/after this time (naive = UTC).")
    ] = None,
) -> OverviewStats:
    overview = await backend.overview(TraceFilters(since=since))
    return OverviewStats.model_validate(overview)


@api.get(
    "/pricing",
    response_model=PricingTableModel,
    tags=["pricing"],
    summary="Active pricing table",
    description="The per-model billing rates (USD per million tokens) tokenlens is "
    "currently using to cost spans.",
)
async def pricing() -> PricingTableModel:
    return PricingTableModel.from_table_dict(get_default_table().as_dict())


@ws.websocket("/ws/live")
async def live_traces(websocket: WebSocket) -> None:
    """Push a TraceSummaryModel JSON message for each trace as it's flushed.

    Best-effort: messages are dropped rather than queued unboundedly if the
    client can't keep up.
    """
    await websocket.accept()
    broadcaster = websocket.app.state.broadcaster
    queue = broadcaster.subscribe()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: send on a socket the client already closed.
        pass
    finally:
        broadcaster.unsubscribe(queue)
