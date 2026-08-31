from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_evaluation import _offline_evaluation

from clothdojo.evaluation import collection
from clothdojo.evaluation.collection import score_evaluation_condition
from clothdojo.evaluation.protocol import load_evaluation_protocol
from clothdojo.evaluation.results import load_evaluation_condition


def test_condition_reader_rejects_rescore_between_metadata_and_results(tmp_path: Path, monkeypatch):
    condition, original_path, templates, _ = _offline_evaluation(tmp_path, quarter_turn=True)
    score_evaluation_condition(condition, original_path, templates)
    updated = replace(load_evaluation_protocol(original_path), symmetry="planar_90")
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()))
    real_open = Path.open
    rescored = False

    def rescore_before_open(path, *args, **kwargs):
        nonlocal rescored
        if path == condition / "results.jsonl" and not rescored:
            rescored = True
            score_evaluation_condition(
                condition, updated_path, templates, rescore_from=original_path
            )
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", rescore_before_open)
    with pytest.raises(RuntimeError, match="changed"):
        load_evaluation_condition(condition)
    assert rescored
    actual = load_evaluation_condition(condition).meta["protocol"]["fingerprint"]
    assert actual == updated.fingerprint


def test_condition_reader_rejects_publication_in_progress_after_initial_check(
    tmp_path: Path, monkeypatch
):
    condition, original_path, templates, _ = _offline_evaluation(tmp_path)
    score_evaluation_condition(condition, original_path, templates)
    real_open = Path.open
    publishing = False

    def begin_publication(path, *args, **kwargs):
        nonlocal publishing
        if path == condition / "results.jsonl" and not publishing:
            publishing = True
            meta_path = condition / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta.update(status="running", phase="publishing_scores")
            meta_path.write_text(json.dumps(meta))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", begin_publication)
    with pytest.raises(RuntimeError, match="changed"):
        load_evaluation_condition(condition)
    assert publishing


def test_concurrent_rescore_cannot_replace_results_under_another_protocol(tmp_path, monkeypatch):
    condition, original_path, templates, original_score = _offline_evaluation(
        tmp_path, quarter_turn=True
    )
    original = load_evaluation_protocol(original_path)
    updated = replace(original, symmetry="planar_90")
    updated_path = tmp_path / "updated.yaml"
    updated_path.write_text(json.dumps(updated.to_json_dict()))
    real_publish = collection.atomic_write_json
    attempted = False

    def rescore_before_publish(path, meta):
        nonlocal attempted
        if meta.get("status") == "complete" and not attempted:
            attempted = True
            with pytest.raises(RuntimeError, match="scoring is already running"):
                score_evaluation_condition(
                    condition, updated_path, templates, rescore_from=original_path
                )
        real_publish(path, meta)

    monkeypatch.setattr(collection, "atomic_write_json", rescore_before_publish)
    score_evaluation_condition(condition, original_path, templates)
    loaded = load_evaluation_condition(condition)
    assert attempted
    assert loaded.meta["protocol"]["fingerprint"] == original.fingerprint
    assert loaded.cases[0].terminal_rigid_distance == original_score.result.terminal_rigid_distance
    lock_inode = (condition / ".scoring.lock").stat().st_ino
    # A later writer can acquire the released lock and keeps the same inode.
    score_evaluation_condition(condition, updated_path, templates, rescore_from=original_path)
    assert (condition / ".scoring.lock").stat().st_ino == lock_inode
    loaded = load_evaluation_condition(condition)
    assert loaded.meta["protocol"]["fingerprint"] == updated.fingerprint
    assert loaded.cases[0].terminal_rigid_distance < original_score.result.terminal_rigid_distance
