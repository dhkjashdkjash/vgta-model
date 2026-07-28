from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    source_count: int
    evolution_count: int
    target_count: int
    steps_per_day: int
    temporal_mode: str
    vim_epochs: int
    gta_epochs: int
    gkoa_population: int
    gkoa_iterations: int


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    formula: str
    unit: str


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return raw


def load_experiment(path: Path) -> ExperimentConfig:
    config = ExperimentConfig(**_read_yaml(path))
    if config.steps_per_day != 6:
        raise ValueError("the manuscript configuration requires six four-hour steps per day")
    if config.temporal_mode != "continuous_multiday":
        raise ValueError("the manuscript configuration requires continuous multi-day sequences")
    return config


def load_feature_schema(path: Path) -> tuple[FeatureSpec, ...]:

    raw = _read_yaml(path)
    rows = raw.get("features")
    if not isinstance(rows, list):
        raise ValueError("feature schema must contain a 'features' list")
    specs = tuple(FeatureSpec(**row) for row in rows)
    if len(specs) != 20:
        raise ValueError(f"feature schema must contain 20 variables, found {len(specs)}")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("feature names must be unique")
    return specs
