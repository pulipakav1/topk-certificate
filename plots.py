"""Plot corrected runs individually, preserving seed and protocol separation."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corrected_results import load_corrected, summarize


def plot_results(paths, out_dir):
    table = summarize(load_corrected(paths))
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    for run_id, rows in table.groupby("run_id"):
        rows = rows.sort_values("hop")
        first = rows.iloc[0]
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))

        for name in ("global", "pairwise", "ordered"):
            column = f"{name}_certificate_rate"
            if column in rows:
                axes[0].plot(rows.hop, rows[column], marker="o", label=name)
        axes[0].set(xlabel="Hop", ylabel="Certificate rate", ylim=(-.02, 1.02))
        axes[0].legend()

        if "ndcg_at_k" in rows:
            axes[1].plot(rows.hop, rows.ndcg_at_k, marker="o")
        axes[1].set(xlabel="Hop", ylabel=f"NDCG@{int(first.retrieval_k)}")

        # Slack: how much room the pairwise decision had, against the boundary margin.
        if "pairwise_gap" in rows:
            axes[2].plot(rows.hop, rows.pairwise_gap, marker="o", label="pairwise slack")
        if "topk_margin" in rows:
            axes[2].plot(rows.hop, rows.topk_margin, marker="s", label="boundary margin")
        axes[2].axhline(0, color="black", linewidth=.8)
        axes[2].set(xlabel="Hop", ylabel="Score units")
        axes[2].legend()

        # Bound tightness: the certified bound against the change that happened.
        if "score_bound" in rows:
            axes[3].plot(rows.hop, rows.score_bound, marker="o", label="bound $B_{max}$")
        if "empirical_score_delta" in rows:
            axes[3].plot(rows.hop, rows.empirical_score_delta, marker="s", label="observed change")
        axes[3].set(xlabel="Hop", ylabel="Score units", yscale="log")
        axes[3].legend()

        fig.suptitle(f"{first.run_kind}: {first.dataset} / {first.retriever}\n"
                     f"seed={first.seed}, noise={first.noise_std}, {first.protocol_version}", fontsize=9)
        fig.tight_layout()
        fig.savefig(directory / f"{run_id}.png", dpi=160)
        plt.close(fig)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True, help="New output directory")
    args = parser.parse_args()
    plot_results(args.input, args.out_dir)


if __name__ == "__main__":
    main()
