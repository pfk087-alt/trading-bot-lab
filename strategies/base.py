"""Contrato base de estrategias e infraestructura de parametros."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


class StrategyError(ValueError):
    """Estrategia o parametros invalidos."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = ()
    help: str = ""

    def coerce(self, value: Any) -> Any:
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "si", "sí"}
            return bool(value)
        if self.kind == "choice":
            if value not in self.choices:
                raise StrategyError(
                    f"{self.name}: valor {value!r} no valido. Opciones: {self.choices}"
                )
            return value
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyError(f"{self.name}: se esperaba un numero, se recibio {value!r}") from exc
        if not np.isfinite(number):
            raise StrategyError(f"{self.name}: el valor debe ser finito")
        if self.minimum is not None and number < self.minimum:
            raise StrategyError(f"{self.name}: minimo {self.minimum}, recibido {number}")
        if self.maximum is not None and number > self.maximum:
            raise StrategyError(f"{self.name}: maximo {self.maximum}, recibido {number}")
        if self.kind == "int":
            if number != int(number):
                raise StrategyError(f"{self.name}: se esperaba un entero")
            return int(number)
        return float(number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
            "help": self.help,
        }


def build_signal_frame(
    index: pd.Index,
    *,
    long_entry: pd.Series | bool | None = None,
    short_entry: pd.Series | bool | None = None,
    long_exit: pd.Series | bool | None = None,
    short_exit: pd.Series | bool | None = None,
) -> pd.DataFrame:
    """Construye un DataFrame de senales booleanas alineado y sin NaN."""
    length = len(index)
    columns = {
        "long_entry": long_entry,
        "short_entry": short_entry,
        "long_exit": long_exit,
        "short_exit": short_exit,
    }
    frame = pd.DataFrame(index=index)
    for name, value in columns.items():
        if value is None:
            frame[name] = False
        elif isinstance(value, pd.Series):
            frame[name] = value.reindex(index).fillna(False).astype(bool)
        else:
            frame[name] = bool(value)
        if len(frame[name]) != length:
            raise StrategyError(f"Longitud incorrecta en la senal {name}")
    conflicting = frame["long_entry"] & frame["short_entry"]
    frame.loc[conflicting, ["long_entry", "short_entry"]] = False
    return frame[["long_entry", "short_entry", "long_exit", "short_exit"]]


class Strategy(ABC):
    """Estrategia de senales. No conoce costes ni tamano de posicion."""

    key: str = "base"
    name: str = "Base"
    description: str = ""
    params: tuple[ParameterSpec, ...] = ()

    def __init__(self, **values: Any) -> None:
        specs = {spec.name: spec for spec in self.params}
        unknown = set(values) - set(specs)
        if unknown:
            raise StrategyError(f"Parametros desconocidos para {self.key}: {sorted(unknown)}")
        resolved: dict[str, Any] = {}
        for spec in self.params:
            raw = values.get(spec.name, spec.default)
            resolved[spec.name] = spec.coerce(raw)
        self.parameters: dict[str, Any] = resolved
        self.validate_parameters()

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {spec.name: spec.default for spec in cls.params}

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return list(cls.params)

    def validate_parameters(self) -> None:
        """Hook para validar combinaciones de parametros."""

    def param(self, name: str) -> Any:
        return self.parameters[name]

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Devuelve senales calculadas solo con informacion hasta cada barra.

        Las senales de la barra t se ejecutan en la apertura de t+1. La
        implementacion no debe usar datos futuros.
        """

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise StrategyError(f"Faltan columnas de datos: {sorted(missing)}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise StrategyError("El indice de datos debe ser DatetimeIndex")
        if data.empty:
            raise StrategyError("No hay datos para generar senales")
        if not data.index.is_monotonic_increasing:
            raise StrategyError("Los datos deben estar ordenados cronologicamente")


@dataclass
class StrategyInfo:
    key: str
    name: str
    description: str
    parameter_specs: list[ParameterSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "parameters": [spec.to_dict() for spec in self.parameter_specs],
        }


def expand_parameter_grid(
    specs: Iterable[ParameterSpec],
    grid: dict[str, Iterable[Any]],
) -> list[dict[str, Any]]:
    """Expande una rejilla de parametros y valida cada combinacion contra las specs."""
    by_name = {spec.name: spec for spec in specs}
    unknown = set(grid) - set(by_name)
    if unknown:
        raise StrategyError(f"Parametros de rejilla desconocidos: {sorted(unknown)}")
    if not grid:
        return [{}]

    names = list(grid)
    value_lists: list[list[Any]] = []
    for name in names:
        spec = by_name[name]
        values = list(grid[name])
        if not values:
            raise StrategyError(f"La rejilla de {name} esta vacia")
        value_lists.append([spec.coerce(value) for value in values])

    combinations: list[dict[str, Any]] = [{}]
    for name, values in zip(names, value_lists):
        combinations = [{**base, name: value} for base in combinations for value in values]
    return combinations
