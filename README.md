# Certifying Top-k Retrieval Stability under Multi-Agent Communication

Agents each hold a bias-free bilinear retrieval head `W` scoring a query against a
candidate passage as

    s_W(q, d) = (Wq)^T (Wd)

A consensus step blends each agent's head with the average of its incoming
neighbours' heads, proposing `W' = W + DeltaW`. The research question is whether a
certificate computed *before* that step, from only the current head, the proposed
update and the embedding norms, can guarantee the query's top-k result is unchanged
after it.

## The three certificates

Each candidate `i` gets a score-change bound from submultiplicativity of the
operator 2-norm:

    B_i = ||q||_2 ||d_i||_2 [ 2 ||W||_2 ||DeltaW||_2 + ||DeltaW||_2^2 ]

- **Pairwise membership**: `s_i - s_j > B_i + B_j` for every `i` in the current
  top-k and every `j` outside it. This certifies the top-k **set** only, not the
  order within it.
- **Global**: the weaker sufficient condition `m_k > 2 B_max`, where `m_k` is the
  boundary margin. Global implies pairwise.
- **Ordered**: the pairwise membership condition **and** the internal
  adjacent-rank conditions `s_(r) - s_(r+1) > B_(r) + B_(r+1)` for
  `r = 1, ..., k-1`. Ordered implies pairwise. Pairwise does **not** imply
  ordered: scores `[3, 3, 1]` with zero bounds and `k = 2` are certified for
  membership and rejected for ordering.

## Install

Requires Python 3.10 or newer; tested on 3.14.2.

    python -m venv .venv
    .venv/Scripts/python.exe -m pip install -e ".[retrieval,reporting,test]"

To reproduce the exact tested environment instead:

    .venv/Scripts/python.exe -m pip install -r requirements.txt
    .venv/Scripts/python.exe -m pip install -e . --no-deps

## Test

    .venv/Scripts/python.exe -B -m pytest -q --hypothesis-show-statistics
    .venv/Scripts/python.exe -B verify_invariants.py
    .venv/Scripts/python.exe -m ruff check .

`pytest` collects `tests/` and `verify_invariants.py`. An autouse fixture fails any
test that leaves a file behind in `runs/`, `results/` or `figures/`.

## Smoke run

A tiny offline consensus run with persisted audit data and no downloads:

    .venv/Scripts/python.exe -B phase1_smoke.py

A small run on real cached data, writing to a scratch directory:

    .venv/Scripts/python.exe -B evaluation_harness.py --pipeline topk-stability \
      --dataset hotpotqa --retriever sentence-transformers/all-MiniLM-L6-v2 \
      --n-eval-queries 2 --candidate-count 4 --retrieval-k 2 --hops 1 \
      --run-kind smoke --out-dir /tmp/smoke

## Settings that are easy to confuse

| Flag | Meaning |
| --- | --- |
| `--graph-neighbors` | Incoming neighbour count: each receiving agent averages at most this many senders. Saturates at `n_agents - 1`. |
| `--retrieval-k` | Retrieval evaluation and certificate depth `k`. |
| `--candidate-count` | Candidate passages per query, one agent per candidate. |
| `--diffusion-alpha` | Consensus blend rate. `--alpha-sweep` overrides it. |
| `--hops` | Number of consensus hops; one certificate decision per query per hop. |
| `--noise-std` | Must be zero for certificate pipelines; nonzero is rejected before any load. |

Retrieval ties are broken by descending score then ascending candidate-slot ID,
and graph neighbour ties by ascending agent index. Neither depends on the seed.

## Paper runs

A `paper` run is gated on reproducible provenance and will refuse to start
otherwise. It requires a clean working tree at a known commit, an explicit
`--model-revision` and `--dataset-revision`, an encoder that actually loaded, and
`noise_std = 0`. Because there is one revision slot per kind, a paper run pins one
dataset and one retriever per invocation.

Every run writes a fresh timestamped directory under `--out-dir` containing a
schema-4 `manifest.json`, the frozen embeddings and initial heads (`inputs.npz`),
the assembled records, per-hop pre-update parameters/scores/bounds and decisions,
`metrics.csv`, `per_query.csv`, `.audit.json` sidecars, and `completed.json` with a
hash of every file. Nothing is overwritten; existing files cause the run to fail.

`validate_run` re-checks the paper claim against the manifest the run wrote, so a
run is paper evidence only if its own artifacts prove the provenance.

### Revisions to pin

These are the commit hashes of the snapshots this repository was developed against.

| Repository | Revision |
| --- | --- |
| `hotpotqa/hotpot_qa` | `1908d6afbbead072334abe2965f91bd2709910ab` |
| `framolfese/2WikiMultihopQA` | `fe713bfbd1afbca1a65246741a75890405d56a3a` |
| `dgslibisey/MuSiQue` | `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `intfloat/e5-small-v2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |

### Main sweep

One invocation per dataset x retriever, nine in total, each sweeping alpha over
five hops. Substitute the revisions from the table above.

    .venv/Scripts/python.exe -B evaluation_harness.py --pipeline full-sweep \
      --dataset <dataset> --retriever <retriever> \
      --dataset-revision <dataset sha> --model-revision <model sha> \
      --n-eval-queries 989 --candidate-count 10 --retrieval-k 5 --graph-neighbors 3 \
      --alpha-sweep 0.1,0.2,0.4,0.6,0.8 --hops 5 --seed 42 --noise-std 0 \
      --run-kind paper --save-per-query --out-dir runs/paper/main

### Candidate-pool ablation

One invocation per dataset, three in total. Omitting `--candidate-count` uses the
10/20/50 ablation; the retriever is fixed to MiniLM.

    .venv/Scripts/python.exe -B evaluation_harness.py --pipeline candidate-ablation \
      --dataset <dataset> \
      --dataset-revision <dataset sha> \
      --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
      --n-eval-queries 989 --retrieval-k 5 --graph-neighbors 3 --diffusion-alpha 0.6 \
      --hops 5 --seed 42 --noise-std 0 --run-kind paper --out-dir runs/paper/ablation

## Reading results

Result CSVs are evidence only through their provenance:

    .venv/Scripts/python.exe -B analyze_results.py --input <run>/metrics.csv --out-dir <new dir>
    .venv/Scripts/python.exe -B plots.py --input <run>/metrics.csv --out-dir <new dir>
    .venv/Scripts/python.exe -B artifact_index.py --root . --output <new file.json>

`--input` takes several paths, so a whole sweep reads at once:

    .venv/Scripts/python.exe -B analyze_results.py \
      --input runs/paper/main/*/*/metrics.csv --out-dir runs/paper/analysis/main

PowerShell does not expand globs for external programs; run these from Git Bash, or
expand first with `(Get-ChildItem -Recurse -Filter metrics.csv runs/paper/main).FullName`.

`analyze_results.py` writes `summary.csv`, one descriptive row per run and hop.
The columns cover retrieval quality (`ndcg_at_k`, `f1_at_k`, `recall_at_k`), the
three certificate rates against what actually changed (`topk_changed_rate`,
`ordering_changed_rate`), the slack each decision had (`pairwise_gap`, `rank_gap`,
`topk_margin`), bound tightness (`score_bound`, `empirical_score_delta`,
`empirical_to_bound_ratio`), per-candidate bound checks (`bound_holds_rate`,
`max_bound_holds_rate`), head movement (`param_change_norm`,
`param_disagreement`), and the three soundness counters, which must all be zero.
`plots.py` draws the same four groups per run.

Aggregates only; `summarize` refuses `per_query.csv` because query-hop rows are
dependent observations. Use the per-query CSVs for per-decision inspection, not
for inferential statistics.

Sweep invocations also write `full_sweep_results.csv`, `certificate_comparison.csv`
and `summary_table.csv` in the parent directory, each with an `.audit.json`
sidecar; `load_corrected` reads them the same way.

`corrected_results.load_corrected` verifies the CSV's hash against its
`.audit.json` sidecar, verifies each cited manifest's hash, revalidates the whole
run, and compares every supplied value against the hashed canonical artifact. A
CSV without that chain is readable only with `allow_legacy=True` and is marked
`legacy_unverified`.

## What is not current evidence

- Everything under `results/`, `figures/` and `protocol_corrected_run.log` predates
  the Phase 1/2 audit. It is retained evidence with unresolved provenance and is
  not authorized for paper claims. `audit.md` records why.
- `abstract.tex` quotes numbers from that pre-audit run; a source comment marks
  them pending revalidation.
- The `runs/` directories labelled `development_fixture` by `artifact_index.py`
  were left by earlier test runs. They pass their hash checks but never loaded an
  encoder, so their HotpotQA/MiniLM metadata describes handcrafted low-dimensional
  vectors, not retrieval.
- `PHASE1.md` is a historical note describing superseded behaviour.

## Layout

| File | Role |
| --- | --- |
| `evaluation_harness.py` | Configs, dataset preparation, embedding, graph construction, the certificate loop, CSV writers, CLI. Also holds the historical synthetic and Fisher/SIR pipelines. |
| `topk_stability.py` | `full_certificate`, `validate_certificate`, `CertificateDecision`, bound helpers, aggregation. |
| `numerical_bounds.py` | Outward-rounded float64 enclosures and the derivation behind them. |
| `dataset_loaders.py` | The three datasets and candidate-pool assembly with the gold-preservation rules. |
| `model_adapters.py` | Per-retriever formatting, pooling, cached-metadata inspection. |
| `run_provenance.py` | Manifests, artifact hashing, git state, `validate_run`, CSV audit sidecars. |
| `corrected_results.py` | Strict reader binding CSV rows to hashed canonical runs. |
| `artifact_index.py` | Read-only inventory of what each artifact establishes. |
| `analyze_results.py`, `plots.py` | Descriptive summaries and per-run plots from audited CSVs. |
| `verify_invariants.py` | Manual invariant checks, also collected by pytest. |
| `CLAUDE.md`, `AGENTS.md` | Working rules and invariants for coding agents. |
| `audit.md` | The Phase 1/2 audit; historical record of findings. |
