from __future__ import annotations

import json
import subprocess
import sys


def test_stage1_shards_cover_every_trial_once() -> None:
    all_ids = []
    for shard in range(10):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.list_scratch_stage1_shard",
                "--config",
                "configs/sweeps/scratch_stage1_random.json",
                "--shard",
                str(shard),
                "--shards",
                "10",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        ids = completed.stdout.splitlines()
        assert len(ids) == 4
        all_ids.extend(ids)
    assert len(all_ids) == 40
    assert len(set(all_ids)) == 40
    assert sorted(all_ids) == [f"stage1-{index:03d}" for index in range(1, 41)]


def test_external_failure_writer_preserves_frozen_config(tmp_path) -> None:
    output = tmp_path / "stage1-001.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.record_scratch_trial_failure",
            "--trial-config",
            "configs/sweeps/scratch_stage1_random.json",
            "--trial-id",
            "stage1-001",
            "--output",
            str(output),
            "--exit-code",
            "124",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_type"] == "timeout"
    assert payload["config"]["trial_id"] == "stage1-001"
