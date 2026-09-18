import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import evaluation_harness as harness
from corrected_results import load_corrected, summarize
from run_provenance import git_state


@pytest.fixture
def result_run(tmp_path):
    cfg = harness.TopKStabilityConfig(n_agents=3, proj_dim=2, k=1, n_hops=1,
        run_dir=str(tmp_path), run_kind="smoke", lm_checkpoint="offline-fixture")
    records = [{"relevance": np.array([1., 0., 0.])}]
    shards = [[{"e_q": torch.tensor([1., 0.]), "e_doc": torch.tensor([x, .5])}]
              for x in (1., .5, -.5)]
    states = [{"weight": torch.eye(2) * (1 + j * .01)} for j in range(3)]
    history = harness.run_topk_stability_experiment(cfg, records, shards, states)
    return next(tmp_path.glob("*/metrics.csv")), history


def test_altered_external_csv_is_rejected(result_run, tmp_path):
    original, _ = result_run
    altered = tmp_path / "altered.csv"
    frame = pd.read_csv(original)
    frame["pairwise_certificate_rate"] = 123.
    frame.to_csv(altered, index=False)
    with pytest.raises(ValueError):
        load_corrected([altered])


@pytest.mark.parametrize("artifact", ["metrics.csv", "per_query.csv", "hop_1_pre_update.npz",
                                      "manifest.json", "hop_1_decisions.json"])
def test_modified_run_artifacts_are_rejected(result_run, artifact):
    original, _ = result_run
    path = original.parent / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_corrected([original])


@pytest.mark.parametrize("artifact", ["hop_1_scores_new.npy", "manifest.json", "completed.json"])
def test_missing_run_artifacts_are_rejected(result_run, artifact):
    original, _ = result_run
    (original.parent / artifact).rename(original.parent / (artifact + ".absent"))
    with pytest.raises(ValueError):
        load_corrected([original])


def test_valid_canonical_and_export_results(result_run, tmp_path):
    original, history = result_run
    frame = load_corrected([original])
    exported = harness.save_full_sweep_csv([harness._topk_dict(m) for m in history], str(tmp_path))
    assert len(load_corrected([exported])) == len(frame) == 1
    summary = summarize(frame)
    assert "pairwise_certificate_violations" in summary
    assert "bound_holds_rate" in summary
    manifest = json.loads((original.parent / "manifest.json").read_text())
    # The manifest records the tree as it is; a clean checkout is not dirty.
    assert manifest["git"]["commit"] and manifest["git"]["dirty"] is git_state()["dirty"]


def test_run_tree_can_be_moved(result_run, tmp_path):
    """Exports cite their manifest relative to themselves, so a run tree relocates."""
    original, _ = result_run
    with original.open(newline="") as stream:
        cited = {row["manifest_path"] for row in csv.DictReader(stream)}
    assert cited == {"manifest.json"}, cited
    # The run_id is the directory name, so a move keeps that name.
    moved = tmp_path / "archive" / original.parent.name
    shutil.copytree(original.parent, moved)
    assert len(load_corrected([moved / "metrics.csv"])) == 1


def test_moved_run_still_fails_on_a_changed_artifact(result_run, tmp_path):
    original, _ = result_run
    moved = tmp_path / "tampered" / original.parent.name
    shutil.copytree(original.parent, moved)
    (moved / "hop_1_decisions.json").write_bytes((moved / "hop_1_decisions.json").read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_corrected([moved / "metrics.csv"])


def test_sweep_export_cites_each_child_run_relatively(result_run, tmp_path):
    original, history = result_run
    exported = Path(harness.save_full_sweep_csv([harness._topk_dict(m) for m in history],
                                                str(original.parent.parent)))
    with exported.open(newline="") as stream:
        cited = {row["manifest_path"] for row in csv.DictReader(stream)}
    assert cited == {f"{original.parent.name}/manifest.json"}, cited
    assert len(load_corrected([exported])) == 1


def test_modified_export_is_rejected(result_run, tmp_path):
    _, history = result_run
    path = harness.save_full_sweep_csv([harness._topk_dict(m) for m in history], str(tmp_path))
    frame = pd.read_csv(path)
    frame["ndcg_at_k"] = .123
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_corrected([path])


def test_changed_in_memory_metrics_cannot_be_summarized(result_run):
    original, _ = result_run
    frame = load_corrected([original])
    frame["ndcg_at_k"] = .123
    with pytest.raises(ValueError):
        summarize(frame)


def test_legacy_reading_requires_explicit_unaudited_mode(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text("hop,ndcg_at_5\n1,0.5\n")
    with pytest.raises(ValueError):
        load_corrected([path])
    frame = load_corrected([path], allow_legacy=True)
    assert frame.attrs["authority"] == "legacy_unverified"
    with pytest.raises(ValueError):
        summarize(frame)
