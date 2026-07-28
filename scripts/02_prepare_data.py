from vgta.commands import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("prepare", description="Pair OLCI products and prepare model inputs"))

