"""Motor de simulacion."""

from .causality import CausalityError, verify_causality
from .costs import CostModel, CostBreakdown
from .engine import BacktestEngine
from .models import BacktestError, BacktestResult, PendingOrder, Position, Trade

__all__ = [
    "BacktestEngine",
    "BacktestError",
    "BacktestResult",
    "CausalityError",
    "CostModel",
    "CostBreakdown",
    "PendingOrder",
    "Position",
    "Trade",
    "verify_causality",
]
