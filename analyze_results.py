"""Analyze explicitly selected corrected aggregate CSVs; never infer legacy provenance."""

import argparse
from pathlib import Path

from corrected_results import load_corrected, summarize


def load_data(paths):
    return load_corrected(paths)


def section1_main_results(full):
    return summarize(full)


def analyze(paths, out_dir):
    frame = load_data(paths)
    table = section1_main_results(frame)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    table.to_csv(directory / "summary.csv", index=False)
    with open(directory / "SUMMARY.md", "x", encoding="utf-8") as stream:
        stream.write("# Corrected-run descriptive summary\n\n")
        stream.write(f"Protocol: {frame.protocol_version.iloc[0]}. Run kind: {frame.run_kind.iloc[0]}.\n\n")
        stream.write(f"{frame.run_id.nunique()} distinct runs; {len(frame)} input rows.\n\n")
        stream.write("Groups retain seed, retrieval depth, noise, dataset, retriever, pool size, "
                     "graph neighbors, alpha, hop count and run ID. No across-seed uncertainty "
                     "or paper claims are inferred. See summary.csv for the measured values.\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True, help="New output directory")
    args = parser.parse_args()
    analyze(args.input, args.out_dir)


if __name__ == "__main__":
    main()
