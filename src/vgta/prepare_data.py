from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vgta.config import load_experiment
from vgta.contracts import LakeSplit, validate_lake_split


DOMAINS = ("source", "evolution", "target")
KINDS = ("features", "concentration", "class", "satellite_mask", "series")


def validate_prepared_npz(
    path: Path,
    *,
    require_paper_counts: bool = True,
    steps_per_day: int = 6,
) -> dict:
    path = Path(path)
    if steps_per_day < 1:
        raise ValueError("steps_per_day must be positive")
    data = np.load(path)
    required = {f"{domain}_{kind}" for domain in DOMAINS for kind in KINDS}
    required.update(f"{domain}_lake" for domain in DOMAINS)
    required.update(("num_classes", "bloom_label"))
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError("prepared data are missing arrays: " + ", ".join(missing))
    shapes = {}
    lake_groups = {}
    num_classes = int(np.asarray(data["num_classes"]).item())
    bloom_label = int(np.asarray(data["bloom_label"]).item())
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if bloom_label < 0 or bloom_label >= num_classes:
        raise ValueError("bloom_label is outside the declared class range")
    for domain in DOMAINS:
        features = data[f"{domain}_features"]
        concentration = data[f"{domain}_concentration"]
        classes = data[f"{domain}_class"]
        satellite_mask = data[f"{domain}_satellite_mask"].astype(bool)
        series = data[f"{domain}_series"]
        lakes = data[f"{domain}_lake"].astype(str)
        if features.ndim != 5 or features.shape[2] != 20:
            raise ValueError(f"{domain}_features must have shape [L,D,20,H,W]")
        lakes_count, days = features.shape[:2]
        if concentration.shape != (lakes_count, days):
            raise ValueError(f"{domain}_concentration must have shape [L,D]")
        if classes.shape != (lakes_count, days):
            raise ValueError(f"{domain}_class must have shape [L,D]")
        if np.any((classes[satellite_mask] < 0) | (classes[satellite_mask] >= num_classes)):
            raise ValueError(f"{domain}_class is outside the declared class range")
        if satellite_mask.shape != (lakes_count, days):
            raise ValueError(f"{domain}_satellite_mask must have shape [L,D]")
        if np.any(~np.isfinite(concentration[satellite_mask])):
            raise ValueError(f"{domain}_concentration is missing on an available satellite day")
        if series.ndim != 2 or series.shape != (lakes_count, days * steps_per_day):
            raise ValueError(
                f"{domain}_series must contain {steps_per_day} four-hour steps for every declared day"
            )
        if lakes.shape != (lakes_count,):
            raise ValueError(f"{domain}_lake must contain one identifier per lake sequence")
        lake_groups[domain] = tuple(sorted(set(lakes.tolist())))
        shapes[domain] = {
            "features": list(features.shape),
            "concentration": list(concentration.shape),
            "series": list(series.shape),
        }
    validate_lake_split(
        LakeSplit(
            source=lake_groups["source"],
            evolution=lake_groups["evolution"],
            target=lake_groups["target"],
        )
    )
    lake_counts = {domain: len(lake_groups[domain]) for domain in DOMAINS}
    if require_paper_counts and tuple(lake_counts[name] for name in DOMAINS) != (18, 4, 5):
        raise ValueError("paper configuration requires exactly 18/4/5 unique lakes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "shapes": shapes,
        "lake_counts": lake_counts,
        "num_classes": num_classes,
        "bloom_label": bloom_label,
    }


def run_preparation(args) -> int:
    if args.input is None or Path(args.input).suffix.lower() != ".npz":
        raise ValueError(
            "preparation requires --input prepared.npz from the authorized study arrays "
            "and product manifest"
        )
    experiment = load_experiment(args.config)
    report = validate_prepared_npz(
        Path(args.input),
        require_paper_counts=True,
        steps_per_day=experiment.steps_per_day,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared_data_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0
