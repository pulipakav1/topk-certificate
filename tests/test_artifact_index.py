"""The inventory must label what each artifact actually establishes."""

import json
from pathlib import PurePath

import numpy as np
import pytest
import torch

import artifact_index
import evaluation_harness as harness


@pytest.fixture
def inventory_root(tmp_path):
    """One fixture run (no encoder) under a runs/ root, plus retained evidence."""
    runs = tmp_path / "runs"
    cfg = harness.TopKStabilityConfig(n_agents=3, proj_dim=2, k=1, n_hops=1,
                                      run_dir=str(runs), run_kind="smoke",
                                      lm_checkpoint="offline-fixture")
    records = [{"relevance": np.array([1., 0., 0.])}]
    shards = [[{"e_q": torch.tensor([1., 0.]), "e_doc": torch.tensor([x, .5])}]
              for x in (1., .5, -.5)]
    states = [{"weight": torch.eye(2) * (1 + j * .01)} for j in range(3)]
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "old_metrics.csv").write_text("hop,ndcg_at_5\n1,0.5\n", encoding="utf-8")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "old_figure.pdf").write_bytes(b"retained")
    return tmp_path


def by_authority(index):
    return {entry["authority"]: entry for entry in index["artifacts"]}


def test_a_run_without_a_loaded_encoder_is_labelled_a_fixture(inventory_root):
    index = artifact_index.build_index(inventory_root)
    entries = by_authority(index)
    assert "development_fixture" in entries
    assert "corrected_smoke" not in entries
    fixture = entries["development_fixture"]
    assert fixture["integrity_verified"] and not fixture["encoder_loaded"]
    assert not any(entry["paper_claims_allowed"] for entry in index["artifacts"])


def test_a_run_that_loaded_its_encoder_keeps_its_run_kind(inventory_root):
    manifest_path = next((inventory_root / "runs").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["encoder_loaded"] = True
    assert artifact_index.run_authority(manifest, verified=True) == "corrected_smoke"
    manifest["run_kind"] = "development"
    assert artifact_index.run_authority(manifest, verified=True) == "corrected_experimental"
    assert artifact_index.run_authority(manifest, verified=False) == "unverified_run"


def test_retained_results_and_figures_stay_legacy_unverified(inventory_root):
    index = artifact_index.build_index(inventory_root)
    legacy = [e for e in index["artifacts"] if e["authority"] == "legacy_unverified"]
    assert {PurePath(e["path"]).as_posix() for e in legacy} == {
        "results/old_metrics.csv", "figures/old_figure.pdf"}
    assert all(e["sha256"] for e in legacy)


def test_a_sweep_parent_is_a_command_record_not_a_run(inventory_root):
    parent = inventory_root / "runs"
    cfg = harness.TopKStabilityConfig(run_kind="development")
    harness._write_configuration(cfg, str(parent), datasets=["hotpotqa"])
    document = json.loads((parent / "sweep_configuration.json").read_text(encoding="utf-8"))
    assert document["document"] == "sweep_configuration"
    assert document["model_revision"] is None and document["dataset_revision"] is None
    assert document["configuration"]["graph_neighbors"] == cfg.baseline_top_k
    assert document["configuration"]["retrieval_k"] == cfg.k
    entries = by_authority(artifact_index.build_index(inventory_root))
    assert entries["sweep_parent"]["paper_claims_allowed"] is False


def test_sweep_configuration_records_requested_revisions(tmp_path):
    cfg = harness.TopKStabilityConfig(run_kind="paper", model_revision="m1", dataset_revision="d1")
    harness._write_configuration(cfg, str(tmp_path))
    document = json.loads((tmp_path / "sweep_configuration.json").read_text(encoding="utf-8"))
    assert document["model_revision"] == "m1" and document["dataset_revision"] == "d1"
    assert document["git"]["commit"]
    with pytest.raises(FileExistsError):
        harness._write_configuration(cfg, str(tmp_path))
