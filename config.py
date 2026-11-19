"""Carga y validacion de configuracion basada en YAML.

Las credenciales nunca deben almacenarse aqui. El laboratorio no envia ordenes
reales, por lo que no necesita claves de broker.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class ConfigError(ValueError):
    """Configuracion invalida."""


@dataclass(frozen=True)
class AppMetaConfig:
    name: str = "TRADING BOT LAB"
    base_currency: str = "USD"


@dataclass(frozen=True)
class AccountConfig:
    initial_capital: float = 10_000.0
    leverage: float = 30.0


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 20.0
    max_open_positions: int = 1
    max_notional_exposure_pct: float = 300.0
    close_on_max_drawdown: bool = False


@dataclass(frozen=True)
class CostConfig:
    spread_pips: float = 1.0
    commission_per_lot_per_side: float = 3.5
    slippage_pips: float = 0.2


@dataclass(frozen=True)
class ExecutionConfig:
    stop_mode: str = "atr"
    stop_value: float = 2.0
    take_profit_mode: str = "rr"
    take_profit_value: float = 2.0
    trailing_stop_atr_mult: float = 0.0
    max_bars_in_trade: int = 0
    warmup_bars: int = 250
    close_at_end: bool = True
    atr_period: int = 14


@dataclass(frozen=True)
class BacktestConfig:
    default_timeframe: str = "1h"
    annualization_days: int = 252
    min_trades_for_ranking: int = 30
    max_optimization_combinations: int = 200


@dataclass(frozen=True)
class StorageConfig:
    database: str = "storage/trading_bot_lab.db"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/trading_bot_lab.log"


@dataclass(frozen=True)
class AppConfig:
    app: AppMetaConfig = field(default_factory=AppMetaConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def database_path(self) -> Path:
        return resolve_project_path(self.storage.database)

    @property
    def log_path(self) -> Path:
        return resolve_project_path(self.logging.file)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"No existe el archivo de configuracion: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalido en {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"El archivo {path} debe contener un objeto YAML.")
    return data


def _build(cls, data: dict[str, Any] | None):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Se esperaba un objeto para {cls.__name__}.")
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Claves desconocidas en {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Carga configuracion YAML y valida rangos criticos."""
    candidate = path or os.environ.get("TBL_CONFIG") or DEFAULT_CONFIG_PATH
    data = _load_yaml(resolve_project_path(candidate))

    config = AppConfig(
        app=_build(AppMetaConfig, data.get("app")),
        account=_build(AccountConfig, data.get("account")),
        risk=_build(RiskConfig, data.get("risk")),
        costs=_build(CostConfig, data.get("costs")),
        execution=_build(ExecutionConfig, data.get("execution")),
        backtest=_build(BacktestConfig, data.get("backtest")),
        storage=_build(StorageConfig, data.get("storage")),
        logging=_build(LoggingConfig, data.get("logging")),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    checks = [
        (config.account.initial_capital > 0, "initial_capital debe ser > 0"),
        (config.account.leverage > 0, "leverage debe ser > 0"),
        (0 < config.risk.risk_per_trade_pct <= 10, "risk_per_trade_pct debe estar en (0, 10]"),
        (0 <= config.risk.max_daily_loss_pct <= 100, "max_daily_loss_pct debe estar en [0, 100]"),
        (0 < config.risk.max_drawdown_pct <= 100, "max_drawdown_pct debe estar en (0, 100]"),
        (config.risk.max_open_positions >= 1, "max_open_positions debe ser >= 1"),
        (config.risk.max_notional_exposure_pct > 0, "max_notional_exposure_pct debe ser > 0"),
        (config.costs.spread_pips >= 0, "spread_pips no puede ser negativo"),
        (config.costs.commission_per_lot_per_side >= 0, "commission no puede ser negativa"),
        (config.costs.slippage_pips >= 0, "slippage no puede ser negativo"),
        (config.execution.stop_mode in {"atr", "pips", "percent", "none"},
         "stop_mode invalido"),
        (config.execution.take_profit_mode in {"rr", "atr", "pips", "percent", "none"},
         "take_profit_mode invalido"),
        (config.execution.warmup_bars >= 0, "warmup_bars no puede ser negativo"),
        (config.execution.atr_period >= 2, "atr_period debe ser >= 2"),
        (config.backtest.annualization_days > 0, "annualization_days debe ser > 0"),
        (config.backtest.max_optimization_combinations >= 1,
         "max_optimization_combinations debe ser >= 1"),
    ]
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)


def default_config_dict() -> dict[str, Any]:
    return AppConfig().to_dict()
