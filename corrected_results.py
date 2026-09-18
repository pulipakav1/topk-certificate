"""Strict readers: provenance is required; historical CSVs are never inferred."""

import json
from pathlib import Path

import pandas as pd

from run_provenance import PROTOCOL_VERSION, cited_manifest, digest, validate_run

GROUP_KEYS = ["protocol_version", "run_kind", "seed", "retrieval_k", "noise_std", "dataset",
              "retriever", "candidate_slots", "graph_neighbors", "diffusion_alpha", "n_hops", "run_id"]
REQUIRED = GROUP_KEYS + ["manifest_path", "hop"]

# Descriptive columns carried into a summary, all of them already recorded per hop.
# Retrieval quality, then the three certificate rates and what actually changed, then
# the slack the decision had, then how loose the bound was against the realized
# change, then how far the heads moved, then the soundness counters.
SUMMARY_METRICS = (
    "ndcg_at_k", "f1_at_k", "recall_at_k",
    "global_certificate_rate", "pairwise_certificate_rate", "ordered_certificate_rate",
    "topk_changed_rate", "ordering_changed_rate",
    "topk_margin", "pairwise_gap", "rank_gap",
    "score_bound", "empirical_score_delta", "empirical_to_bound_ratio",
    "bound_holds_rate", "max_bound_holds_rate",
    "param_change_norm", "param_disagreement",
    "global_certificate_violations", "pairwise_certificate_violations",
    "ordered_certificate_violations",
)


def resolve_manifest_paths(frame, csv_dir):
    """Absolute manifest_path in memory, however the CSV cited it.

    Exports cite their manifest relative to themselves so a run tree can be moved;
    an older export citing an absolute path still resolves, because joining an
    absolute path discards the base.
    """
    frame = frame.copy()
    frame["manifest_path"] = [str(cited_manifest(name, csv_dir).resolve())
                              for name in frame.manifest_path]
    return frame


def validate_frame(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Legacy/unverified artifact: missing provenance columns {sorted(missing)}")
    if frame.empty or frame[REQUIRED].isna().any().any():
        raise ValueError("Empty or incomplete provenance is not corrected evidence")
    if set(frame.protocol_version) != {PROTOCOL_VERSION}:
        raise ValueError("Legacy or mixed protocol versions cannot be aggregated")
    if frame.run_kind.nunique() != 1:
        raise ValueError("Smoke, development and paper results must be analyzed separately")
    for run_id, group in frame.groupby("run_id", sort=False):
        if group[GROUP_KEYS + ["manifest_path"]].nunique().max() > 1:
            raise ValueError("One run ID has conflicting protocol/configuration metadata")
        path = Path(group.manifest_path.iloc[0])
        if not path.is_file():
            raise ValueError(f"Missing manifest for run {run_id}")
        manifest = validate_run(path)
        if manifest.get("run_id") != run_id or manifest.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Run ID/protocol does not match manifest")
        config = manifest["configuration"]
        expected = {
            "seed": manifest["seed"], "retrieval_k": config["retrieval_k"],
            "noise_std": config["noise_std"], "dataset": config["dataset"],
            "retriever": config["lm_checkpoint"], "candidate_slots": config["candidate_slots"],
            "graph_neighbors": config["graph_neighbors"], "diffusion_alpha": config["alpha"],
            "n_hops": config["hop_count"], "run_kind": manifest["run_kind"],
        }
        for key, expected_value in expected.items():
            if not group[key].eq(expected_value).all():
                raise ValueError(f"CSV metadata disagrees with manifest: {key}")
        if config["noise_std"] != 0:
            raise ValueError("Noisy runs cannot be certificate evidence")
        per_query = "query_index" in group
        canonical_path = path.parent / ("per_query.csv" if per_query else "metrics.csv")
        canonical = resolve_manifest_paths(
            pd.read_csv(canonical_path, float_precision="round_trip"), canonical_path.parent)
        row_key = ["hop", "query_index"] if per_query else ["hop"]
        if group.duplicated(row_key).any():
            raise ValueError("Duplicate run/hop rows")
        extras = set(group.columns) - set(canonical.columns)
        if extras - {"n_candidates"}:
            raise ValueError(f"Columns absent from canonical artifact: {sorted(extras)}")
        if "n_candidates" in extras and not group.n_candidates.eq(config["candidate_slots"]).all():
            raise ValueError("Candidate count disagrees with canonical artifact")
        columns = [c for c in group.columns if c in canonical.columns and c not in row_key]
        actual = group.set_index(row_key).sort_index()[columns]
        try:
            expected_rows = canonical.set_index(row_key).loc[actual.index, columns]
            pd.testing.assert_frame_equal(actual, expected_rows, check_dtype=False, check_exact=True)
        except (KeyError, AssertionError) as exc:
            raise ValueError("Result values disagree with hashed canonical artifact") from exc
    return frame


def load_corrected(paths, *, allow_legacy=False):
    frames = []
    legacy = []
    for filename in paths:
        path = Path(filename)
        try:
            frame = pd.read_csv(path, float_precision="round_trip")
            is_legacy = "protocol_version" not in frame or set(frame.protocol_version) != {PROTOCOL_VERSION}
            legacy.append(is_legacy)
            if is_legacy:
                if not allow_legacy:
                    raise ValueError("Legacy/unverified artifact; explicit allow_legacy=True is required")
                frames.append(frame)
                continue
            audit = json.loads(Path(str(path) + ".audit.json").read_text(encoding="utf-8"))
            if audit["protocol_version"] != PROTOCOL_VERSION or audit["sha256"] != digest(path):
                raise ValueError("Consumed result artifact hash mismatch")
            if set(frame.manifest_path) != set(audit["manifests"]):
                raise ValueError("Result/manifest audit mismatch")
            for name, sha in audit["manifests"].items():
                if digest(cited_manifest(name, path.parent)) != sha:
                    raise ValueError("Consumed result manifest hash mismatch")
            frames.append(validate_frame(resolve_manifest_paths(frame, path.parent)))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Missing or invalid result audit: {path}") from exc
    if not frames:
        raise ValueError("At least one explicit corrected CSV path is required")
    if any(legacy):
        if not all(legacy):
            raise ValueError("Legacy and audited runs cannot be combined")
        combined = pd.concat(frames, ignore_index=True)
        combined.attrs["authority"] = "legacy_unverified"
        return combined
    combined = pd.concat(frames, ignore_index=True)
    validate_frame(combined)
    row_key = ["run_id", "hop"] + (["query_index"] if "query_index" in combined else [])
    if combined.duplicated(row_key).any():
        raise ValueError("Duplicate run/hop rows; refuse to count the same result twice")
    combined.attrs["authority"] = "audited"
    combined.attrs["sources"] = {str(Path(p).resolve()): digest(p) for p in paths}
    return combined


def summarize(frame):
    validate_frame(frame)
    if "query_index" in frame:
        raise ValueError("Summaries require aggregate metrics.csv; query-hop rows are dependent observations")
    metrics = [name for name in SUMMARY_METRICS if name in frame]
    if not metrics:
        raise ValueError("Expected aggregate corrected metrics using *_at_k columns")
    return frame.groupby(GROUP_KEYS + ["hop"], as_index=False, dropna=False)[metrics].mean()
