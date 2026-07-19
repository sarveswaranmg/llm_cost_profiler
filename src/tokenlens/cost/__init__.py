"""Cost computation and attribution.

Maps model + token usage to a dollar cost using a versioned pricing table
(pricing_data.json), and aggregates cost across traces by user, feature,
model, node, or kind.
"""

from tokenlens.cost.attribution import (
    CostBreakdownEntry,
    cost_by,
    retry_waste,
    trace_flame_data,
)
from tokenlens.cost.pricing import (
    ModelPricing,
    PricingTable,
    compute_cost,
    get_default_table,
    load_pricing,
)

__all__ = [
    "CostBreakdownEntry",
    "ModelPricing",
    "PricingTable",
    "compute_cost",
    "cost_by",
    "get_default_table",
    "load_pricing",
    "retry_waste",
    "trace_flame_data",
]
