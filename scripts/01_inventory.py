from vgta.commands import dispatch


if __name__ == "__main__":
    raise SystemExit(dispatch("inventory", description="Inventory source data and validate the lake split"))

