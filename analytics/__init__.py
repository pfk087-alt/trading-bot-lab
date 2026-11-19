"""Estadistica, evaluacion y validacion de estrategias."""

from .metrics import (
    compute_metrics,
    compute_monthly_returns,
    metrics_to_frame,
    objective_score,
)

__all__ = [
    "compute_metrics",
    "compute_monthly_returns",
    "metrics_to_frame",
    "objective_score",
]
