"""The historical Fisher/SIR writers must not overwrite retained evidence."""

from pathlib import Path

import pytest

import evaluation_harness as harness

WRITERS = [
    ("save_results_csv", "routing_comparison_metrics.csv"),
    ("save_hotpotqa_results_csv", "hotpotqa_real_metrics.csv"),
    ("save_theorem1_diagnostics_csv", "hotpotqa_theorem1_diagnostics.csv"),
    ("save_margin_diagnostics_csv", "hotpotqa_margin_diagnostics.csv"),
]


def synthetic_history():
    return [harness.HopMetrics(hop=1, lambda2=.5, ndcg_at_k=.5, f1_at_k=.5, mean_param_drift=.1)]


def hotpot_history():
    return [harness.HotpotHopMetrics(hop=1, lambda2=.5, ndcg_at_k=.5, f1_at_k=.5, cumulative_d_ig=.1)]


def history_for(writer):
    return synthetic_history() if writer == "save_results_csv" else hotpot_history()


@pytest.mark.parametrize("writer,filename", WRITERS)
def test_legacy_writers_refuse_to_replace_an_existing_file(writer, filename, tmp_path):
    (tmp_path / filename).write_text("retained evidence\n", encoding="utf-8")
    history = history_for(writer)
    with pytest.raises(FileExistsError):
        getattr(harness, writer)(history, history, out_dir=str(tmp_path))
    assert (tmp_path / filename).read_text(encoding="utf-8") == "retained evidence\n"


@pytest.mark.parametrize("writer,filename", WRITERS)
def test_legacy_writers_write_into_a_given_directory(writer, filename, tmp_path):
    history = history_for(writer)
    path = getattr(harness, writer)(history, history, out_dir=str(tmp_path))
    assert path == str(tmp_path / filename)
    assert (tmp_path / filename).is_file()


def test_legacy_figure_writers_refuse_to_replace_existing_figures(tmp_path):
    (tmp_path / "hotpotqa_scaling.pdf").write_bytes(b"retained figure")
    with pytest.raises(FileExistsError):
        harness.plot_hotpotqa_figure(hotpot_history(), hotpot_history(), 5, out_dir=str(tmp_path))
    assert (tmp_path / "hotpotqa_scaling.pdf").read_bytes() == b"retained figure"

    (tmp_path / "routing_comparison_trends.pdf").write_bytes(b"retained figure")
    with pytest.raises(FileExistsError):
        harness.plot_ieee_figures(synthetic_history(), synthetic_history(), 5, out_dir=str(tmp_path))
    assert (tmp_path / "routing_comparison_trends.pdf").read_bytes() == b"retained figure"


def test_legacy_default_directory_is_fresh_and_outside_results(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = Path(harness.legacy_output_dir("synthetic"))
    second = Path(harness.legacy_output_dir("synthetic"))
    assert first != second
    for directory in (first, second):
        assert directory.parent == tmp_path / "runs" / "legacy"
        assert "results" not in directory.parts and "figures" not in directory.parts
