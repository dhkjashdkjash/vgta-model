from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LakeSplit:

    source: tuple[str, ...]
    evolution: tuple[str, ...]
    target: tuple[str, ...]


def validate_lake_split(split: LakeSplit) -> None:
    """Reject duplicate or overlapping lake identifiers."""
    groups = (split.source, split.evolution, split.target)
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("lake split contains a duplicate name within a group")
    sets = tuple(set(group) for group in groups)
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("lake split contains overlap between groups")
