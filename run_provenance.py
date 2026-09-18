"""Versioned, offline provenance and artifact authority for certificate runs."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import csv
import os
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

PROTOCOL_VERSION = "topk-fp64-audited"
CANDIDATE_POLICY = {
    "identity": "zero-based candidate slot; fixed across hops",
    "passage_identity": "(title, text); paragraphs sharing a title are distinct passages",
    "support": "per paragraph where the source gives it (MuSiQue is_supporting), else by "
               "supporting title (HotpotQA, 2Wiki)",
    "deduplication": "byte-identical repeats collapse; a repeat whose support flag disagrees "
                     "raises ValueError",
    "gold": "all referenced supports required; overflow past the candidate count raises ValueError",
    "distractors": "seeded same-query shuffle then cross-query non-support padding",
    "padding_deduplication": "exclude chosen/gold titles and take one passage per padded title",
    "ranking": "descending score, ascending slot ID; stable sort",
    "evaluation": "candidate-pool reranking, not corpus retrieval",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    # JSON has no infinities; keep the mathematical sentinel explicitly textual.
    def clean(x):
        if hasattr(x, "tolist"):
            return clean(x.tolist())
        if isinstance(x, float) and not __import__("math").isfinite(x):
            return str(x)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        return x
    with open(path, "x", encoding="utf-8") as file:
        json.dump(clean(data), file, indent=2, allow_nan=False)


def git_state():
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=Path(__file__).parent,
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "status": status, "diff_sha256": hashlib.sha256((git("diff", "HEAD") or "").encode()).hexdigest()}


def start_run(cfg, records):
    import numpy as np
    import torch
    from model_adapters import adapter_metadata

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex
    directory = Path(cfg.run_dir or "runs") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    config = asdict(cfg)
    normalized = {**config, "graph_neighbors": cfg.baseline_top_k, "retrieval_k": cfg.k,
                  "candidate_slots": cfg.n_agents, "hop_count": cfg.n_hops, "alpha": cfg.diffusion_alpha}
    source = dict(records[0].get("source_metadata", {})) if records else {}
    dataset = {"name": cfg.dataset, "split": cfg.dataset_slice or source.get("split"),
               "requested_revision": cfg.dataset_revision, "revision": source.get("revision"),
               "fingerprint": source.get("fingerprint"), "provenance_available": bool(source)}
    # Raw-file sources (MuSiQue) also record which file was read and its hash.
    dataset.update({key: source[key] for key in ("repository", "source_file", "source_sha256", "loader")
                    if key in source})
    adapter = adapter_metadata(cfg.lm_checkpoint, max_seq_len=cfg.max_seq_len,
                               revision=cfg.model_revision)
    model = {"checkpoint": cfg.lm_checkpoint, "requested_revision": cfg.model_revision,
             "revision": cfg.resolved_model_revision, "adapter": adapter,
             "encoder_loaded": cfg.resolved_model_revision is not None}
    packages = {d.metadata["Name"]: d.version for d in distributions() if d.metadata.get("Name")}
    manifest = {
        "schema_version": 4, "protocol_version": PROTOCOL_VERSION, "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_kind": cfg.run_kind,
        "command": {"executable": sys.executable, "argv": sys.argv,
                    "display": subprocess.list2cmdline([sys.executable, *sys.argv])},
        "configuration": normalized, "seed": cfg.seed, "dataset": dataset, "model": model,
        "packages": packages, "python": sys.version, "numpy": np.__version__, "pytorch": torch.__version__,
        "platform": {"system": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "numerics": {"device": "cpu", "dtype": "float64", "encoder_device": cfg.device,
                     "encoder_dtype": "float32", "score_kernel": "sequential multiply-add binary64",
                     "certificate": "outward spectral upper bound + old/new rounding envelope; exact comparisons"},
        "git": git_state(), "candidate_pool_policy": CANDIDATE_POLICY, "query_count": len(records),
        "source_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(directory / "manifest.json", manifest)
    # Keep the Phase 1 audit filename for existing consumers; both hold this manifest.
    write_json(directory / "configuration.json", manifest)
    write_json(directory / "records.json", records)
    return directory, manifest


def finish_run(directory):
    directory = Path(directory)
    write_json(directory / "completed.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    })


def cited_manifest(name, csv_dir):
    """Resolve a manifest_path cell. Relative cells are read next to their CSV."""
    return Path(csv_dir) / name


def _relative_to_csv(name, csv_dir):
    try:
        return Path(os.path.relpath(name, csv_dir)).as_posix()
    except ValueError:
        return name  # different drive on Windows; keep the absolute path


def record_csv(path):
    """Bind an export's bytes to its source manifests; canonical rows are checked on read.

    Each cited manifest is rewritten relative to the CSV that cites it, so moving a
    run tree keeps the citation valid. The hash is taken after that rewrite, and the
    manifest digests are unchanged.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames, rows = reader.fieldnames, list(reader)
    if any(row["manifest_path"] != _relative_to_csv(row["manifest_path"], path.parent) for row in rows):
        for row in rows:
            row["manifest_path"] = _relative_to_csv(row["manifest_path"], path.parent)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    manifests = sorted({row["manifest_path"] for row in rows})
    write_json(str(path) + ".audit.json", {
        "sha256": digest(path), "protocol_version": PROTOCOL_VERSION,
        "manifests": {name: digest(cited_manifest(name, path.parent)) for name in manifests},
    })


def validate_run(manifest_path):
    """Verify the complete set of artifacts required by the current run schema."""
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        complete = json.loads((path.parent / "completed.json").read_text(encoding="utf-8"))
        if manifest["protocol_version"] != PROTOCOL_VERSION or manifest["schema_version"] != 4:
            raise ValueError("Legacy/unverified run protocol")
        if manifest["run_id"] != path.parent.name:
            raise ValueError("Manifest/run directory mismatch")
        files = complete["files"]
        required = {"manifest.json", "configuration.json", "records.json", "inputs.npz",
                    "metrics.csv", "per_query.csv", "metrics.csv.audit.json", "per_query.csv.audit.json"}
        hops = manifest["configuration"]["hop_count"]
        for hop in range(1, hops + 1):
            required.update({f"hop_{hop}_pre_update.npz", f"hop_{hop}_scores_new.npy",
                             f"hop_{hop}_decisions.json"})
        if not required <= files.keys():
            raise ValueError(f"Missing required artifact hashes: {sorted(required - files.keys())}")
        for name, sha in files.items():
            artifact = path.parent / name
            if (Path(name).name != name or artifact.resolve().parent != path.parent or
                    not artifact.is_file() or digest(artifact) != sha):
                raise ValueError(f"Run artifact integrity failure: {name}")
        configuration = json.loads((path.parent / "configuration.json").read_text(encoding="utf-8"))
        if configuration != manifest:
            raise ValueError("Manifest/configuration mismatch")
        if manifest["run_kind"] == "paper":
            # The launch gate runs before the artifacts exist. Re-check the claim
            # against what the run actually recorded, so a paper run is only paper
            # evidence if its own manifest proves the provenance.
            paper_requirements = {
                "clean working tree": manifest["git"]["dirty"] is False,
                "known commit": bool(manifest["git"]["commit"]),
                "requested model revision": bool(manifest["model"]["requested_revision"]),
                "resolved model revision": bool(manifest["model"]["revision"]),
                "loaded encoder": manifest["model"]["encoder_loaded"] is True,
                "requested dataset revision": bool(manifest["dataset"]["requested_revision"]),
                "dataset provenance": manifest["dataset"]["provenance_available"] is True,
                "zero noise": manifest["configuration"]["noise_std"] == 0,
            }
            unmet = sorted(name for name, met in paper_requirements.items() if not met)
            if unmet:
                raise ValueError(f"Run claims run_kind=paper without: {unmet}")
        for filename, per_query in (("metrics.csv", False), ("per_query.csv", True)):
            with (path.parent / filename).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            keys = [(int(row["hop"]), int(row["query_index"])) if per_query else int(row["hop"])
                    for row in rows]
            expected = ({(hop, query) for hop in range(1, hops + 1)
                         for query in range(manifest["query_count"])} if per_query else set(range(1, hops + 1)))
            if set(keys) != expected or len(keys) != len(expected):
                raise ValueError(f"Incomplete or duplicate run rows: {filename}")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid run artifacts: {path}") from exc
