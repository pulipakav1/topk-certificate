"""The descriptive summary and plots must carry the recorded certificate columns."""

import numpy as np
import pytest
import torch

import evaluation_harness as harness
from corrected_results import SUMMARY_METRICS, load_corrected, summarize
import analyze_results
import plots


@pytest.fixture
def audited_run(tmp_path):
    cfg = harness.TopKStabilityConfig(n_agents=4, proj_dim=2, k=2, n_hops=2,
                                      run_dir=str(tmp_path / "runs"), run_kind="smoke",
                                      lm_checkpoint="offline-fixture")
    records = [{"relevance": np.array([1., 0., 1., 0.])}]
    shards = [[{"e_q": torch.tensor([1., 0.]), "e_doc": torch.tensor([x, 0.])}]
              for x in (1., .7, .2, -.3)]
    states = [{"weight": torch.eye(2) * (1 + j * .01)} for j in range(4)]
    harness.run_topk_stability_experiment(cfg, records, shards, states)
    return next((tmp_path / "runs").glob("*/metrics.csv"))


def test_summary_reports_slack_and_bound_tightness(audited_run):
    table = summarize(load_corrected([audited_run]))
    for name in ("pairwise_gap", "rank_gap", "topk_margin", "score_bound",
                 "empirical_score_delta", "empirical_to_bound_ratio",
                 "param_change_norm", "param_disagreement"):
        assert name in table, name
    assert len(table) == 2  # one row per hop
    assert (table.pairwise_certificate_violations == 0).all()


def test_every_summary_metric_is_a_recorded_column():
    """A summary column must exist in what a run actually writes."""
    recorded = set(harness._TOPK_METRIC_COLUMNS)
    assert set(SUMMARY_METRICS) <= recorded, sorted(set(SUMMARY_METRICS) - recorded)


def test_analyze_writes_a_new_directory_and_refuses_to_reuse_it(audited_run, tmp_path):
    out = tmp_path / "analysis"
    table = analyze_results.analyze([audited_run], str(out))
    assert (out / "summary.csv").is_file() and (out / "SUMMARY.md").is_file()
    assert "pairwise_gap" in table
    with pytest.raises(FileExistsError):
        analyze_results.analyze([audited_run], str(out))


def test_plots_writes_one_figure_per_run(audited_run, tmp_path):
    out = tmp_path / "figures"
    table = plots.plot_results([audited_run], str(out))
    written = list(out.glob("*.png"))
    assert len(written) == table.run_id.nunique() == 1
    assert written[0].stat().st_size > 0
    with pytest.raises(FileExistsError):
        plots.plot_results([audited_run], str(out))


def test_query_hop_rows_are_not_summarized(audited_run):
    per_query = audited_run.parent / "per_query.csv"
    frame = load_corrected([per_query])
    with pytest.raises(ValueError, match="dependent observations"):
        summarize(frame)
