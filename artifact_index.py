"""Inventory authority without modifying historical files. No dataset/model I/O."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from run_provenance import PROTOCOL_VERSION, digest, write_json, validate_run


def run_authority(manifest, verified):
    """Label a run by what its own manifest establishes.

    A fixture run carries the default HotpotQA/MiniLM configuration but never
    loaded an encoder, because it was built from handcrafted low-dimensional
    vectors. Its metadata therefore reads like a retrieval run and is not one, so
    it gets its own label rather than being pooled with real development runs.
    """
    if not verified:
        return "unverified_run"
    if not manifest.get("model", {}).get("encoder_loaded"):
        return "development_fixture"
    return "corrected_smoke" if manifest.get("run_kind") == "smoke" else "corrected_experimental"


def build_index(root):
    root = Path(root).resolve()
    entries = []
    for folder in ("results", "figures"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                entries.append({"path": str(path.relative_to(root)), "authority": "legacy_unverified",
                                "paper_claims_allowed": False, "sha256": digest(path)})
    for manifest_path in sorted((root / "runs").rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verified = False
        if manifest.get("protocol_version") == PROTOCOL_VERSION:
            try:
                validate_run(manifest_path)
                verified = True
            except ValueError:
                pass
        authority = run_authority(manifest, verified)
        entries.append({"path": str(manifest_path.parent.relative_to(root)),
                        "authority": authority,
                        "run_id": manifest.get("run_id"), "protocol_version": manifest.get("protocol_version"),
                        "run_kind": manifest.get("run_kind"), "integrity_verified": verified,
                        "encoder_loaded": bool(manifest.get("model", {}).get("encoder_loaded")),
                        "paper_claims_allowed": False,
                        "reason": "Separate scientific review required; protocol completion is not paper authorization"})
    for sweep in sorted((root / "runs").rglob("sweep_configuration.json")):
        document = json.loads(sweep.read_text(encoding="utf-8"))
        entries.append({"path": str(sweep.parent.relative_to(root)), "authority": "sweep_parent",
                        "run_kind": document.get("run_kind"), "paper_claims_allowed": False,
                        "sha256": digest(sweep),
                        "reason": "Command record; the per-run manifests inside are the provenance"})
    for config in sorted((root / "runs").rglob("configuration.json")):
        if (config.parent / "manifest.json").exists():
            continue
        entries.append({"path": str(config.parent.relative_to(root)), "authority": "legacy_unverified",
                        "paper_claims_allowed": False, "sha256": digest(config)})
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "corrected_protocol": PROTOCOL_VERSION, "artifacts": entries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True, help="New index path; existing indexes are not overwritten")
    args = parser.parse_args()
    write_json(args.output, build_index(args.root))
