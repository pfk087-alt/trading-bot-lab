"""Persistencia JSON/CSV de sesiones de paper trading (independiente de SQLite de backtests)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .engine import PaperState


def _default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


class PaperSessionStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: PaperState) -> Path:
        path = self.directory / f"{state.session_id}.json"
        path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2, default=_default),
            encoding="utf-8",
        )
        return path

    def load(self, session_id: str) -> PaperState:
        path = self.directory / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No existe la sesion {session_id}")
        return PaperState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_sessions(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            rows.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "started_at": data.get("started_at"),
                    "symbol": data.get("symbol"),
                    "timeframe": data.get("timeframe"),
                    "strategy": data.get("strategy_name"),
                    "bars": data.get("bars_processed"),
                    "equity": data.get("equity"),
                    "initial_capital": data.get("initial_capital"),
                    "trades": len(data.get("trades", [])),
                }
            )
        return pd.DataFrame(rows)

    def export_trades_csv(self, state: PaperState, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.directory / f"{state.session_id}_trades.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(state.trades).to_csv(target, index=False)
        return target
