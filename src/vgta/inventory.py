from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


PRODUCT_RE = re.compile(
    r"^(?P<platform>S3[AB])_OL_(?P<level>[12])_(?P<product_type>[A-Z]{3})_{4}"
    r"(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6})_"
)


@dataclass(frozen=True)
class ProductRecord:
    path: Path
    lake: str
    platform: str
    level: int
    start: str
    end: str
    product_type: str


@dataclass(frozen=True)
class ProductPair:
    level1: ProductRecord
    level2: ProductRecord


@dataclass(frozen=True)
class RejectedGroup:
    lake: str
    platform: str
    start: str
    end: str
    reason: str
    paths: Tuple[str, ...]


def parse_product(path: Path, *, lake: str) -> ProductRecord:

    match = PRODUCT_RE.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized Sentinel-3 product name: {path.name}")
    return ProductRecord(
        path=path,
        lake=lake,
        platform=match.group("platform"),
        level=int(match.group("level")),
        start=match.group("start"),
        end=match.group("end"),
        product_type=match.group("product_type"),
    )


def pair_products(
    records: Iterable[ProductRecord],
) -> Tuple[List[ProductPair], List[RejectedGroup]]:

    grouped = {}
    for record in records:
        key = (record.lake, record.platform, record.start, record.end)
        grouped.setdefault(key, []).append(record)
    pairs: List[ProductPair] = []
    rejected: List[RejectedGroup] = []
    for key in sorted(grouped):
        group = grouped[key]
        level1 = [item for item in group if item.level == 1]
        level2 = [item for item in group if item.level == 2]
        if len(level1) == len(level2) == 1:
            pairs.append(ProductPair(level1=level1[0], level2=level2[0]))
            continue
        reason = "ambiguous pair" if len(level1) > 1 or len(level2) > 1 else "unmatched product"
        rejected.append(
            RejectedGroup(*key, reason=reason, paths=tuple(str(item.path) for item in group))
        )
    return pairs, rejected


def discover_products(root: Path) -> Tuple[List[ProductRecord], List[str]]:
    records: List[ProductRecord] = []
    errors: List[str] = []
    for path in sorted(root.rglob("*.SEN3")):
        try:
            records.append(parse_product(path, lake=path.parent.name))
        except ValueError as exc:
            errors.append(str(exc))
    return records, errors


def run_inventory(args) -> int:
    root = args.data_root or args.input
    if root is None:
        raise ValueError("inventory requires --data-root or --input")
    records, errors = discover_products(Path(root))
    pairs, rejected = pair_products(records)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "products": [{**asdict(row), "path": str(row.path)} for row in records],
        "pairs": [
            {
                "level1": str(pair.level1.path),
                "level2": str(pair.level2.path),
            }
            for pair in pairs
        ],
        "rejected": [asdict(row) for row in rejected],
        "parse_errors": errors,
    }
    (output / "inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0
