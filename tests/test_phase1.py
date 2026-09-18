"""Offline regression tests for the certificate pipeline; no model downloads."""

from dataclasses import FrozenInstanceError, asdict
import csv
import json

import numpy as np
import pytest
import torch

import evaluation_harness as harness
from dataset_loaders import RawExample, _assemble_candidates
from topk_stability import full_certificate, validate_certificate, topk_indices


def tiny_run(run_dir):
    cfg = harness.TopKStabilityConfig(n_agents=4, proj_dim=2, k=2, n_hops=2,
                                      run_dir=str(run_dir))
    records = [{"relevance": np.array([1., 0., 1., 0.])}]
    shards = [[{"e_q": torch.tensor([1., 0.]),
                "e_doc": torch.tensor([x, 0.])}] for x in [1., .7, .2, -.3]]
    states = [{"weight": torch.eye(2) * (1 + j * .01)} for j in range(4)]
    return cfg, records, shards, states


def test_certificate_is_immutable_and_post_scores_cannot_change_it():
    scores = np.array([5., 4., 1., 0.])
    bounds = np.full(4, .1)
    decision = full_certificate(scores, 2, bounds)
    snapshot = asdict(decision)
    stable = validate_certificate(decision, scores.copy())
    changed = validate_certificate(decision, -scores)
    assert not stable["topk_changed"] and changed["topk_changed"]
    assert asdict(decision) == snapshot
    scores[:] = 0
    bounds[:] = 100
    assert asdict(decision) == snapshot
    with pytest.raises(FrozenInstanceError):
        decision.ordered_certified = False
    with pytest.raises(TypeError):
        full_certificate(scores, 2, bounds, scores_new=scores)


def test_tied_boundary_rejects_all_certificates():
    scores = np.array([3., 2., 2., 0.])
    decision = full_certificate(scores, 2, np.zeros(4))
    assert topk_indices(scores, 2).tolist() == [0, 1]
    assert not decision.global_certified
    assert not decision.pairwise_certified
    assert not decision.ordered_certified


def test_internal_tie_rejects_order_but_allows_membership():
    decision = full_certificate(np.array([3., 3., 1.]), 2, np.zeros(3))
    assert decision.global_certified and decision.pairwise_certified
    assert not decision.order_certified and not decision.ordered_certified
    assert decision.topk == (0, 1)


@pytest.mark.parametrize("noise", [.01, -1., float("nan"), float("inf")])
def test_noise_rejected_before_scoring_or_mutation(noise, tmp_path, monkeypatch):
    cfg, records, shards, states = tiny_run(tmp_path)
    cfg.noise_std = noise
    monkeypatch.setattr(harness, "unflatten_into", lambda *a: pytest.fail("mutated head"))
    with pytest.raises(ValueError, match="noise_std=0"):
        harness.run_topk_stability_experiment(cfg, records, shards, states)
    with pytest.raises(ValueError, match="noise_std=0"):
        full_certificate(np.array([2., 1.]), 1, np.zeros(2), noise_std=noise)


def test_gold_overflow_raises_without_consuming_rng():
    example = RawExample("overflow", {"a", "b", "c"}, [(t, t) for t in "abc"])
    rng = np.random.default_rng(42)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="3 gold passages.*candidate_slots=2"):
        _assemble_candidates(example, 2, [], rng)
    assert rng.bit_generator.state == state


def test_gold_exact_capacity_keeps_every_gold():
    example = RawExample("valid", {"a", "b"}, [(t, t) for t in "abc"])
    docs, relevance = _assemble_candidates(example, 2, [], np.random.default_rng(42))
    assert set(docs) == {"a: a", "b: b"}
    assert relevance.tolist() == [1., 1.]


def test_decisions_precede_head_update_and_labels_are_isolated(tmp_path, monkeypatch):
    cfg, records, shards, states = tiny_run(tmp_path)
    records.append({"relevance": records[0]["relevance"].copy()})
    for shard in shards:
        shard.append({key: value.clone() for key, value in shard[0].items()})
    events = []
    certify, apply, validate = harness.full_certificate, harness.unflatten_into, harness.validate_certificate

    def record_decision(*args, **kwargs):
        events.append("decision")
        return certify(*args, **kwargs)

    def record_update(*args, **kwargs):
        events.append("update")
        return apply(*args, **kwargs)

    def record_validation(*args, **kwargs):
        events.append("validation")
        return validate(*args, **kwargs)

    monkeypatch.setattr(harness, "full_certificate", record_decision)
    monkeypatch.setattr(harness, "unflatten_into", record_update)
    monkeypatch.setattr(harness, "validate_certificate", record_validation)
    rows = []
    history = harness.run_topk_stability_experiment(cfg, records, shards, states, rows)
    assert events == (["decision"] * len(records) + ["update"] * cfg.n_agents +
                      ["validation"] * len(records)) * cfg.n_hops
    assert all(row["noise_std"] == 0 and row["retrieval_k"] == cfg.k for row in rows)
    assert all(m.global_certificate_violations == m.pairwise_certificate_violations ==
               m.ordered_certificate_violations == 0 for m in history)
    records[0]["relevance"] = 1 - records[0]["relevance"]
    other_rows = []
    harness.run_topk_stability_experiment(cfg, records, shards, states, other_rows)
    for old, new in zip(rows, other_rows):
        for key in old.keys() - {"ndcg_at_k", "f1_at_k", "recall_at_k", "run_id", "manifest_path"}:
            assert old[key] == new[key], key


@pytest.mark.parametrize("overrides,message", [
    ({"diffusion_alpha": float("nan")}, "diffusion_alpha must be finite"),
    ({"diffusion_alpha": float("inf")}, "diffusion_alpha must be finite"),
    ({"n_hops": 0}, "at least 1"),
    ({"n_eval_queries": 0}, "at least 1"),
    ({"baseline_top_k": -1}, "graph_neighbors must be nonnegative"),
    ({"k": 4}, "0 < k < candidate_slots"),
])
def test_degenerate_settings_are_rejected_before_any_run(overrides, message, tmp_path, monkeypatch):
    cfg, records, shards, states = tiny_run(tmp_path)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    monkeypatch.setattr(harness, "start_run", lambda *a: pytest.fail("started a run directory"))
    with pytest.raises(ValueError, match=message):
        harness.run_topk_stability_experiment(cfg, records, shards, states)


def test_paper_runs_require_revisions_and_a_clean_tree(tmp_path, monkeypatch):
    cfg, records, shards, states = tiny_run(tmp_path)
    cfg.run_kind = "paper"
    with pytest.raises(ValueError, match="model_revision and dataset_revision"):
        harness._validate_topk_config(cfg)
    cfg.model_revision = "abc123"
    cfg.dataset_revision = "def456"
    for state in ({"commit": "abc", "dirty": True}, {"commit": None, "dirty": False},
                  {"commit": "abc", "dirty": None}):
        monkeypatch.setattr(harness, "git_state", lambda state=state: state)
        with pytest.raises(ValueError, match="clean working tree"):
            harness._validate_topk_config(cfg)
    monkeypatch.setattr(harness, "git_state", lambda: {"commit": "abc", "dirty": False})
    harness._validate_topk_config(cfg)


def test_paper_sweeps_cannot_share_one_revision_across_repositories(monkeypatch):
    monkeypatch.setattr(harness, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(harness, "_require_topk_deps", lambda: pytest.fail("loaded model deps"))
    options = {"run_kind": "paper", "model_revision": "m1", "dataset_revision": "d1"}
    with pytest.raises(ValueError, match="one dataset per invocation"):
        harness.main_full_sweep(run_options=options, retrievers=["intfloat/e5-small-v2"])
    with pytest.raises(ValueError, match="one retriever per invocation"):
        harness.main_full_sweep(run_options=options, datasets=["hotpotqa"])
    with pytest.raises(ValueError, match="one dataset per invocation"):
        harness.main_candidate_pool_ablation(run_options=options)


def test_graph_neighbors_limits_incoming_edges_per_receiver():
    # --graph-neighbors is the incoming budget: each receiver averages at most
    # this many neighbours, while a popular agent may send to everyone.
    theta = np.array([[1., 0.], [.99, .1], [.9, .3], [0., 1.]])
    nodes = [f"agent_{j}" for j in range(4)]
    graph = harness.build_baseline_graph(theta, nodes, top_k=2)
    assert max(dict(graph.in_degree()).values()) <= 2
    assert max(dict(graph.out_degree()).values()) > 2


def test_graph_neighbour_ties_break_by_agent_index():
    """Equal similarities resolve by ascending agent index, like retrieval ties.

    The interleaved similarity levels below are a case where an unstable sort
    picks agents 7, 9, 10 instead of 7, 8, 9.
    """
    directions = [(1., 0.), (.8, .6), (.6, .8)]
    levels = [0, 1, 2, 2, 1, 2, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 0, 2, 0, 1,
              2, 0, 1, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    theta = np.array([directions[level] for level in levels])
    nodes = [f"agent_{j}" for j in range(len(levels))]
    graph = harness.build_baseline_graph(theta, nodes, top_k=3)
    senders = sorted(int(u.split("_")[1]) for u, _ in graph.in_edges("agent_0"))
    assert senders == [7, 8, 9]
    again = harness.build_baseline_graph(theta.copy(), nodes, top_k=3)
    assert sorted(graph.edges(data=True)) == sorted(again.edges(data=True))


def test_untied_graph_neighbours_are_unchanged_by_the_tie_rule():
    theta = np.array([[1., 0.], [.99, .1], [.9, .3], [0., 1.]])
    nodes = [f"agent_{j}" for j in range(4)]
    graph = harness.build_baseline_graph(theta, nodes, top_k=2)
    assert sorted(u for u, _ in graph.in_edges("agent_0")) == ["agent_1", "agent_2"]


def test_cli_separates_graph_neighbors_and_retrieval_depth():
    args = harness.parse_args(["--pipeline", "topk-stability", "--graph-neighbors", "1", "--retrieval-k", "2"])
    assert args.top_k == 1 and args.retrieval_k == 2
    with pytest.warns(FutureWarning, match="deprecated"):
        old = harness.parse_args(["--top-k", "4"])
    assert old.top_k == 4 and old.retrieval_k is None
    with pytest.raises(SystemExit):
        harness.parse_args(["--top-k", "4", "--graph-neighbors", "3"])


def test_audit_written_before_mutation_and_all_csvs_record_noise(tmp_path, monkeypatch):
    cfg, records, shards, states = tiny_run(tmp_path)
    apply = harness.unflatten_into

    def check_persisted(*args):
        assert list(tmp_path.glob("*/hop_1_decisions.json"))
        return apply(*args)

    monkeypatch.setattr(harness, "unflatten_into", check_persisted)
    rows = []
    history = harness.run_topk_stability_experiment(cfg, records, shards, states, rows)
    config = json.loads(next(tmp_path.glob("*/configuration.json")).read_text())
    assert config["configuration"]["noise_std"] == 0
    assert config["configuration"]["k"] == 2
    assert len(list(tmp_path.glob("*/hop_*_pre_update.npz"))) == cfg.n_hops
    aggregate = [{"dataset": "tiny", "retriever": "offline", "diffusion_alpha": .6,
                  **harness._topk_dict(m)} for m in history]
    paths = [
        harness.save_topk_stability_csv(history, str(tmp_path)),
        harness.save_alpha_sweep_csv({.6: history}, str(tmp_path)),
        harness.save_full_sweep_csv(aggregate, str(tmp_path)),
        harness.save_certificate_comparison_csv(aggregate, str(tmp_path)),
        harness.save_summary_table_csv(aggregate, str(tmp_path)),
        harness.save_per_query_csv(rows, str(tmp_path)),
        harness.save_candidate_pool_ablation_csv(
            [row | {"n_candidates": 4}
             for row in aggregate], str(tmp_path)),
    ]
    for path in paths:
        with open(path, newline="") as file:
            written = list(csv.DictReader(file))
        assert written
        assert all(float(row["noise_std"]) == 0 and int(row["retrieval_k"]) == 2 for row in written)


@pytest.mark.parametrize("entry", [harness.main_topk_stability, harness.main_full_sweep,
                                  harness.main_candidate_pool_ablation])
def test_noise_fails_before_dataset_or_model_loading(entry, monkeypatch):
    monkeypatch.setattr(harness, "_require_topk_deps", lambda: pytest.fail("requested model dependencies"))
    with pytest.raises(ValueError, match="noise_std=0"):
        entry(noise_std=.1)


def test_tied_metrics_use_candidate_slots(tmp_path):
    cfg, records, shards, states = tiny_run(tmp_path)
    cfg.diffusion_alpha = 0
    cfg.n_hops = 1
    for j in range(cfg.n_agents):
        shards[j][0]["e_doc"] = torch.tensor([1., 0.])
        states[j]["weight"] = torch.eye(2)
    records[0]["relevance"] = np.array([1., 1., 0., 0.])
    history = harness.run_topk_stability_experiment(cfg, records, shards, states)
    assert history[0].recall_at_k == 1
    assert history[0].ndcg_at_k == 1
    assert history[0].pairwise_certificate_rate == 0
