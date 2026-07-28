from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_metric_bundle(
    output: Path,
    *,
    classification: Mapping[str, object],
    reconstruction: Mapping[str, object],
    provenance: Mapping[str, object],
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "metrics.json"
    payload = {
        "classification": dict(classification),
        "reconstruction": dict(reconstruction),
        "provenance": dict(provenance),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def make_paper_outputs(args) -> int:
    if args.input is None:
        raise ValueError("paper output stage requires --input with prediction artifacts")
    from vgta.pipeline import run_paper_stage

    return int(run_paper_stage(args) or 0)
