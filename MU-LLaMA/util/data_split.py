"""Deterministic record- or audio-grouped train/validation/test splits."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, Iterable, List


def record_group(record: dict, index: int, split_unit: str) -> str:
    if split_unit == "record":
        return f"record:{index}"
    if split_unit != "audio":
        raise ValueError("data.split_unit must be 'record' or 'audio'")
    value = (
        record.get("audio_id")
        or record.get("audio_path")
        or record.get("audio_name")
    )
    return f"audio:{value}" if value not in (None, "") else f"record:{index}"


def split_record_indices(
    records: List[dict],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    split_unit: str = "record",
) -> Dict[str, List[int]]:
    """Create seeded splits without allowing an audio group to cross splits.

    [Paper E1: evaluation protocol] Formal paper runs use ``split_unit=audio``
    and a separate external test collection; the test set is never tuned on.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("data.test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    if len(records) < 2:
        raise ValueError("Dataset is too small for the requested split")

    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record_group(record, index, split_unit)].append(index)
    group_names = sorted(grouped)
    required_groups = 3 if test_fraction > 0 else 2
    if len(group_names) < required_groups:
        raise ValueError(
            f"data.split_unit={split_unit!r} produced {len(group_names)} groups; "
            f"at least {required_groups} are required"
        )
    random.Random(int(seed)).shuffle(group_names)

    def take_groups(target_samples: int, remaining_required: int) -> List[str]:
        selected = []
        selected_samples = 0
        while (
            group_names
            and len(group_names) > remaining_required
            and (not selected or selected_samples < target_samples)
        ):
            name = group_names.pop(0)
            selected.append(name)
            selected_samples += len(grouped[name])
        return selected

    total = len(records)
    test_groups = (
        take_groups(max(1, round(total * test_fraction)), remaining_required=2)
        if test_fraction > 0
        else []
    )
    validation_groups = take_groups(
        max(1, round(total * validation_fraction)), remaining_required=1
    )
    training_groups = group_names
    if not training_groups or not validation_groups or (test_fraction > 0 and not test_groups):
        raise ValueError("Dataset groups are too small for non-empty train/validation/test splits")

    def indices(names: Iterable[str]) -> List[int]:
        return sorted(index for name in names for index in grouped[name])

    return {
        "train": indices(training_groups),
        "validation": indices(validation_groups),
        "test": indices(test_groups),
    }


def split_manifest(
    records: List[dict],
    split_indices: Dict[str, List[int]],
    split_unit: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    external_test_records: List[dict] | None = None,
) -> dict:
    groups_by_split = {
        name: sorted({
            record_group(records[index], index, split_unit)
            for index in indices
        })
        for name, indices in split_indices.items()
    }
    if external_test_records is not None:
        groups_by_split["test"] = sorted({
            record_group(record, index, split_unit)
            for index, record in enumerate(external_test_records)
        })
    overlap = {}
    names = ("train", "validation", "test")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap[f"{left}_{right}"] = len(
                set(groups_by_split[left]) & set(groups_by_split[right])
            )
    return {
        "seed": int(seed),
        "split_unit": split_unit,
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "test_source": "external" if external_test_records is not None else "internal",
        "splits": {
            name: {
                "samples": (
                    len(external_test_records)
                    if name == "test" and external_test_records is not None
                    else len(split_indices[name])
                ),
                "groups": len(groups),
                "group_sha256": hashlib.sha256(
                    "\n".join(groups).encode("utf-8")
                ).hexdigest(),
            }
            for name, groups in groups_by_split.items()
        },
        "group_overlap": overlap,
    }
