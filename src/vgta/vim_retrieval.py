from __future__ import annotations


def run_vim(args) -> int:
    if args.input is None:
        raise ValueError("ViM requires --input pointing to prepared NPZ data")
    from vgta.pipeline import run_vim_stage

    return int(run_vim_stage(args) or 0)
