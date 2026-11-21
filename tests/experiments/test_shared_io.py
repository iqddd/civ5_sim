"""Guard artifact bytes and JSONL recovery behavior at the shared I/O boundary."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from civ5rl.experiments._shared.io import (
    append_jsonl,
    atomic_legacy_torch_save,
    read_jsonl,
    read_jsonl_lines,
    recover_jsonl_prefix,
)
from civ5rl.experiments.ppo_artifacts import pack_rollout, save_rollout
from civ5rl.provenance.common import file_sha256


def test_jsonl_keeps_reader_differences(tmp_path: Path) -> None:
    path = tmp_path / "updates.jsonl"
    with pytest.raises(FileNotFoundError):
        read_jsonl(path)
    assert read_jsonl(path, missing_ok=True) == []

    path.write_bytes('{"text":"a\u2028b"}\r\n\r\n{"next":1}'.encode())
    assert read_jsonl(path, streaming=True) == [{"text": "a\u2028b"}, {"next": 1}]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)

    path.write_text('{"next":1}\n \n', encoding="utf-8")
    assert read_jsonl(path) == [{"next": 1}]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, skip_blank=False)

    path.write_text('{"next":1}\n{"next":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_append_jsonl_rejects_nonfinite_rows(tmp_path: Path) -> None:
    path = tmp_path / "updates.jsonl"
    append_jsonl(path, {"text": "Ω", "step": 1})
    assert path.read_text(encoding="utf-8") == '{"text": "\\u03a9", "step": 1}\n'
    before = path.read_bytes()
    with pytest.raises(ValueError):
        append_jsonl(path, {"loss": float("nan")})
    assert path.read_bytes() == before


def test_step_log_recovery_preserves_committed_text(tmp_path: Path) -> None:
    path = tmp_path / "steps.jsonl"
    committed = '{ "step": 1, "labels": 5 }'
    path.write_text(committed + '\n{"step":', encoding="utf-8")
    lines = read_jsonl_lines(path)
    assert recover_jsonl_prefix(path, lines, 1, missing_error="missing steps") == [
        {"step": 1, "labels": 5}
    ]
    assert path.read_text(encoding="utf-8") == committed + "\n"


def test_legacy_torch_save_overwrites_with_identical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "critic-adaptation.pt"
    legacy_path = path.with_suffix(".tmp")
    data = torch.arange(4)
    for count in (1, 3):
        value = {"base": data, "slice": data[:count]}
        torch.save(value, legacy_path)
        expected = legacy_path.read_bytes()
        legacy_path.unlink()
        atomic_legacy_torch_save(value, path)
        assert path.read_bytes() == expected
        assert list(tmp_path.iterdir()) == [path]


def test_legacy_torch_save_leaves_existing_artifact_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    atomic_legacy_torch_save({"update": 1}, path)
    before = path.read_bytes()

    def fail_save(value: object, temporary: Path) -> None:
        temporary.write_bytes(b"partial archive")
        raise OSError("interrupted archive write")

    with (
        patch.object(torch, "save", side_effect=fail_save),
        pytest.raises(OSError, match="interrupted archive write"),
    ):
        atomic_legacy_torch_save({"update": 2}, path)

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_save_rollout_keeps_archive_and_metadata_format(tmp_path: Path) -> None:
    episodes = [{"actions": [("end_turn",)], "chosen": [0], "encoded": None}]
    reference = tmp_path / "first.tmp"
    torch.save(
        {
            "format": "rl01.rollout.pickle",
            "payload": pickle.dumps(pack_rollout(episodes), protocol=5),
        },
        reference,
    )
    record = save_rollout(episodes, tmp_path, 1)
    path = tmp_path / "first.pt"
    assert path.read_bytes() == reference.read_bytes()
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) == record
    timings = record.pop("timings")
    assert list(timings) == [
        "prepare_seconds",
        "serialization_seconds",
        "write_seconds",
        "hash_seconds",
        "total_seconds",
    ]
    assert list(record.items()) == list(
        {
            "update": 1,
            "path": "first.pt",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "episodes": 1,
            "decisions": 1,
            "omitted_reconstructible_fields": [
                "encoded",
                "encoded_variant",
                "old_hidden",
            ],
            "format": "rl01.rollout.v2",
        }.items()
    )
