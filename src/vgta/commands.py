from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Sequence

from vgta.config import load_experiment


STAGE_TARGETS = {
    "inventory": ("vgta.inventory", "run_inventory"),
    "prepare": ("vgta.prepare_data", "run_preparation"),
    "vim": ("vgta.vim_retrieval", "run_vim"),
    "gkoa": ("vgta.gkoa", "run_gkoa"),
    "gta": ("vgta.gta_reconstruction", "run_gta"),
    "paper": ("vgta.paper_outputs", "make_paper_outputs"),
    "all": ("vgta.pipeline", "run_pipeline"),
}


class StageUnavailableError(RuntimeError):
    pass


def build_parser(stage: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="experiment configuration file",
    )
    parser.add_argument("--data-root", type=Path, help="local data directory outside Git")
    parser.add_argument("--input", type=Path, help="stage input directory or manifest")
    parser.add_argument("--output", type=Path, default=Path("results") / stage)
    return parser


def dispatch(stage: str, argv: Sequence[str] | None = None, *, description: str) -> int:
    parser = build_parser(stage, description)
    args = parser.parse_args(argv)
    load_experiment(args.config)

    module_name, function_name = STAGE_TARGETS[stage]
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise StageUnavailableError(f"Stage '{stage}' is unavailable.") from exc

    return int(function(args) or 0)
