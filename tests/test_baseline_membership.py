from __future__ import annotations

from scripts.run_baselines import frozen_tuning_membership


def test_frozen_tuning_membership_is_deterministic_and_stratified() -> None:
    rows = []
    for label in ("flip", "notflip"):
        for index in range(100):
            rows.append(
                {
                    "sample_id": f"{label}-{index:03d}",
                    "label": label,
                }
            )
    first = frozen_tuning_membership(rows, seed=42)
    second = frozen_tuning_membership(rows, seed=42)
    assert first == second
    assert sum(
        value == "validation" for value in first.values()
    ) == 20
    for label in ("flip", "notflip"):
        validation = [
            row
            for row in rows
            if row["label"] == label
            and first[row["sample_id"]] == "validation"
        ]
        assert len(validation) == 10
