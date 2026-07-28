from vgta.commands import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("all", description="Run all processing stages in order"))
