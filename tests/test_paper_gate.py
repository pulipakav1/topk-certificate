"""A run may only call itself `paper` when its provenance is complete."""

import hashlib
import json

import numpy as np
import pytest
import torch

import corrected_results
import evaluation_harness as harness
import run_provenance
from run_provenance import validate_run


def paper_config(tmp_path, **overrides):
    settings = dict(n_agents=4, proj_dim=2, k=2, n_hops=1, run_dir=str(tmp_path),
                    run_kind="paper", model_revision="m1", dataset_revision="d1",
                    resolved_model_revision="resolved-m1")
    settings.update(overrides)
    return harness.TopKStabilityConfig(**settings)


def tiny_inputs():
    # A paper run must come from a real dataset load, so the records carry the
    # source metadata that load_dataset_records attaches.
    records = [{"relevance": np.array([1., 0., 1., 0.]),
                "source_metadata": {"repository": "hotpot_qa", "split": "validation[:1]",
                                    "revision": "d1", "fingerprint": "fp", "configuration": "distractor"}}]
    shards = [[{"e_q": torch.tensor([1., 0.]), "e_doc": torch.tensor([x, 0.])}]
              for x in (1., .7, .2, -.3)]
    states = [{"weight": torch.eye(2) * (1 + j * .01)} for j in range(4)]
    return records, shards, states


@pytest.fixture
def clean_tree(monkeypatch):
    """Both the launch gate and the manifest record must see a clean tree."""
    state = {"commit": "abc123", "dirty": False, "status": "", "diff_sha256": "0" * 64}
    monkeypatch.setattr(harness, "git_state", lambda: dict(state))
    monkeypatch.setattr(run_provenance, "git_state", lambda: dict(state))


def test_dirty_tree_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "git_state", lambda: {"commit": "abc123", "dirty": True})
    with pytest.raises(ValueError, match="clean working tree"):
        harness._validate_topk_config(paper_config(tmp_path))


def test_unknown_commit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "git_state", lambda: {"commit": None, "dirty": False})
    with pytest.raises(ValueError, match="clean working tree"):
        harness._validate_topk_config(paper_config(tmp_path))


@pytest.mark.parametrize("missing", ["model_revision", "dataset_revision"])
def test_missing_revision_is_refused(missing, tmp_path, clean_tree):
    with pytest.raises(ValueError, match="model_revision and dataset_revision"):
        harness._validate_topk_config(paper_config(tmp_path, **{missing: None}))


def test_nonzero_noise_is_refused(tmp_path, clean_tree):
    with pytest.raises(ValueError, match="noise_std=0"):
        harness._validate_topk_config(paper_config(tmp_path, noise_std=.01))


def test_an_unloaded_encoder_is_refused(tmp_path, clean_tree, monkeypatch):
    """A paper run must load the retriever it names, not a handcrafted fixture."""
    cfg = paper_config(tmp_path, resolved_model_revision=None)
    records, shards, states = tiny_inputs()
    monkeypatch.setattr(harness, "start_run", lambda *a: pytest.fail("started a run directory"))
    with pytest.raises(ValueError, match="resolved encoder revision"):
        harness.run_topk_stability_experiment(cfg, records, shards, states)


def test_a_complete_paper_run_is_accepted_and_records_its_provenance(tmp_path, clean_tree):
    cfg = paper_config(tmp_path)
    records, shards, states = tiny_inputs()
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    manifest_path = next(tmp_path.glob("*/manifest.json"))
    manifest = validate_run(manifest_path)
    assert manifest["run_kind"] == "paper"
    assert manifest["git"]["commit"] == "abc123" and manifest["git"]["dirty"] is False
    assert manifest["model"]["revision"] == "resolved-m1"
    assert manifest["model"]["encoder_loaded"] is True
    assert manifest["configuration"]["noise_std"] == 0
    assert manifest["dataset"]["requested_revision"] == "d1"


def test_a_paper_run_with_a_broken_artifact_hash_is_not_readable(tmp_path, clean_tree):
    cfg = paper_config(tmp_path)
    records, shards, states = tiny_inputs()
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    run = next(tmp_path.glob("*/manifest.json")).parent
    (run / "inputs.npz").write_bytes((run / "inputs.npz").read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity failure"):
        validate_run(run / "manifest.json")
    with pytest.raises(ValueError):
        corrected_results.load_corrected([run / "metrics.csv"])


def test_a_missing_completion_record_is_not_readable(tmp_path, clean_tree):
    cfg = paper_config(tmp_path)
    records, shards, states = tiny_inputs()
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    run = next(tmp_path.glob("*/manifest.json")).parent
    (run / "completed.json").unlink()
    with pytest.raises(ValueError, match="Missing or invalid run artifacts"):
        validate_run(run / "manifest.json")


def test_incomplete_hop_coverage_is_not_readable(tmp_path, clean_tree):
    cfg = paper_config(tmp_path, n_hops=2)
    records, shards, states = tiny_inputs()
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    run = next(tmp_path.glob("*/manifest.json")).parent
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["hop_count"] == 2
    lines = (run / "metrics.csv").read_text(encoding="utf-8").splitlines()
    (run / "metrics.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failure|Incomplete"):
        validate_run(run / "manifest.json")


@pytest.mark.parametrize("section,key,value", [
    ("git", "dirty", True),
    ("git", "commit", None),
    ("model", "requested_revision", None),
    ("model", "revision", None),
    ("model", "encoder_loaded", False),
    ("dataset", "requested_revision", None),
    ("dataset", "provenance_available", False),
    ("configuration", "noise_std", .01),
])
def test_a_manifest_claiming_paper_without_provenance_is_refused(
        section, key, value, tmp_path, clean_tree):
    """The recorded claim is re-checked, not only the claim made at launch."""
    cfg = paper_config(tmp_path)
    records, shards, states = tiny_inputs()
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    run = next(tmp_path.glob("*/manifest.json")).parent
    assert validate_run(run / "manifest.json")["run_kind"] == "paper"

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest[section][key] = value
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    for name in ("manifest.json", "configuration.json"):
        (run / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        completed["files"][name] = run_provenance.digest(run / name)
    (run / "completed.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="run_kind=paper without"):
        validate_run(run / "manifest.json")


def test_paper_sweeps_require_one_repository_per_invocation(clean_tree, monkeypatch):
    monkeypatch.setattr(harness, "_require_topk_deps", lambda: pytest.fail("loaded model deps"))
    options = {"run_kind": "paper", "model_revision": "m1", "dataset_revision": "d1"}
    with pytest.raises(ValueError, match="one dataset per invocation"):
        harness.main_full_sweep(run_options=options, retrievers=["intfloat/e5-small-v2"])
    with pytest.raises(ValueError, match="one retriever per invocation"):
        harness.main_full_sweep(run_options=options, datasets=["hotpotqa"])


def test_git_state_reads_a_non_ascii_diff(tmp_path, monkeypatch):
    """git output is UTF-8; decoding it with the Windows code page lost stdout."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "note.md").write_text("plain\n", encoding="utf-8")
    git("add", "note.md")
    git("commit", "-q", "-m", "init")
    # U+201D encodes to e2 80 9d; 0x9d has no cp1252 character.
    (tmp_path / "note.md").write_text("a “quoted” note\n", encoding="utf-8")
    monkeypatch.setattr(run_provenance, "__file__", str(tmp_path / "run_provenance.py"))
    state = run_provenance.git_state()
    assert state["commit"] and state["dirty"] is True
    assert "note.md" in state["status"]
    assert state["diff_sha256"] != hashlib.sha256(b"").hexdigest()
