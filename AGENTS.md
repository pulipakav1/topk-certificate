# AGENTS.md — Certifying Top-k Retrieval Stability under Multi-Agent Communication

## Project purpose

This repository supports the paper **“Certifying Top-k Retrieval Stability under Multi-Agent Communication.”** It studies whether a deterministic one-hop consensus update of agent-specific, bias-free bilinear retrieval heads can change a query’s retrieved top-*k*. Its contribution is a sufficient, predictive certificate computed from the **pre-update** scores, parameters, and proposed parameter update; it is not a claim about general end-to-end retriever training or multi-hop QA accuracy.

Earlier synthetic and HotpotQA Fisher/SIR routing experiments have been removed from the code; they survive only in git history and in the retained `results/`/`figures/` artifacts, which must not be conflated with the paper’s top-*k* certificate pipeline.

## Repository map

- `evaluation_harness.py` — primary entry point. Contains the top-*k* experiment, sweeps, CSV writers, and CLI. The dataset-to-numerical-runner integration has an embedding-shape defect described below.
- `topk_stability.py` — certificate math, top-*k* helpers, score-change bounds, and aggregate diagnostics.
- `dataset_loaders.py` — HotpotQA, MuSiQue, and 2WikiMultiHopQA adapters and candidate-pool construction.
- `model_adapters.py` — checkpoint-specific query/passage formatting and pooling.
- `numerical_bounds.py` — CPU float64 scoring, outward-rounded operator-norm enclosures, and score-roundoff envelopes.
- `run_provenance.py` — per-run manifests, source hashes, pre/post-update artifact provenance, and completion hashes.
- `corrected_results.py`, `artifact_index.py` — result metadata/integrity checks and an inventory separating historical artifacts from current-protocol runs. These checks do not establish scientific validity or authenticate arbitrary copied input CSVs.
- `verify_invariants.py` — 50 manually runnable invariant checks, also explicitly discovered by the pytest configuration in `pyproject.toml`.
- `tests/` — certificate, pipeline, candidate, CLI, and numerical regression/property tests. The pipeline fixtures use one-dimensional embeddings and do not exercise the real preparation boundary.
- `analyze_results.py` — reads explicitly selected current-protocol CSVs through `corrected_results.py`; writes `summary.csv` and `SUMMARY.md` to a new directory.
- `plots.py` — reads explicitly selected current-protocol CSVs and writes one PNG per run to a new directory.
- `graph_connectivity.py` — weighted adjacency and the consensus graph's algebraic connectivity `lambda2` (a reported diagnostic; ARPACK's unseeded start vector can change its last bits between identical runs).
- `results/full_sweep_results.csv`, `results/full_sweep_per_query.csv`, `results/candidate_pool_ablation.csv` — historical artifacts without the metadata required by the current analysis/plot readers; do not mix them with current-protocol runs.
- `results/tables/`, `results/PAPER_RESULTS_SUMMARY.md`, `figures/topk_stability/` — derived artifacts, never primary raw evidence.
- `abstract.tex` — draft paper abstract. The certificate implication chain is corrected; its empirical numbers still come from an unprovenanced pre-audit run and are marked pending revalidation in a source comment.
- `README.md` — install, test, smoke and paper-run commands; `requirements.txt` pins the tested environment.
- Ruff (`E`, `F`, `W`; `E501`/`E741`/`E731` ignored) is configured in `pyproject.toml` and passes. It is a check, not a formatter.
- `PHASE1.md` — historical implementation/verification notes, partly superseded by the current numerical, provenance, and reporting code.
- `audit.md` — technical baseline, reproduced defects, verification results, and deferred work recorded on 2026-09-17.

## Primary top-k pipeline

`evaluation_harness.py --pipeline full-sweep` executes:

1. `load_dataset_records` loads a validation slice and builds one candidate pool per question.
2. `_prepare_hotpotqa_run` formats text for the chosen checkpoint, encodes frozen query/passages, pools, L2-normalizes, and creates one candidate-slot/agent shard per passage.
3. It initializes one near-identity **bias-free** head per agent (`W_j`), using seeded float32 initialization; the numerical runner copies these stored values into float64 heads.
4. At each hop, `build_baseline_graph` creates a directed cosine-neighbor graph from flattened head parameters. This top-*k* pipeline does not use SIR/Fisher routing.
5. Scores before communication are `s_j(q,d_j) = (W_j q)^T(W_j d_j)`, evaluated by an explicit CPU float64 multiply/add sequence.
6. The weighted incoming consensus proposal is formed as `Theta_consensus`; `Delta W_j` is its difference from the old head.
7. Outward-rounded per-document bounds, old/proposed scoring-error envelopes, and immutable global/pairwise/ordered decisions are derived from pre-update scores and this deterministic proposal. Parameter endpoints, old scores, total bounds, embedding norms, and decision JSON are persisted before mutation.
8. The proposal is assigned without noise, post-update scores are computed, and separate metrics/empirical soundness diagnostics are recorded. Each hop is recertified.

`full_certificate(scores_old, k, bounds, ...)` accepts no post-update scores and returns a frozen `CertificateDecision`. `validate_certificate(decision, scores_new)` computes diagnostics without changing the decision. Tests check this order and persistence before mutation. The separate older `topk_stability_certificate(scores_old, scores_new, k)` remains a post-hoc diagnostic and is not used for predictive decisions.

Current integration blocker: `_build_agent_shards` emits `(1, hidden_size)` query/document tensors, while `bilinear_score` and `certified_bounds` require one-dimensional vectors. The unmodified dataset path fails before its first certificate. Passing numerical fixtures do not establish that the dataset/retriever entry points work end to end.

## Mathematical contract

For a single head, the intended score is

`s_W(q,d) = (Wq)^T(Wd)`.

The actual multi-agent experiment applies its own head to each candidate: `s_j(q,d_j) = (W_jq)^T(W_jd_j)`. For candidate `j`, with deterministic `W'_j = W_j + DeltaW_j`, the code uses the operator/spectral 2-norm:

`B_j(q,d_j) = ||q||_2 ||d_j||_2 [2 ||W_j||_2 ||DeltaW_j||_2 + ||DeltaW_j||_2^2]`.

The real-arithmetic helper `spectral_norm` uses `np.linalg.norm(W, ord=2)`. The current experiment instead uses `spectral_norm_upper` and `update_norm_upper`: SVD factors are enclosed using interval Gram matrices and a reconstruction residual to upper-bound the same operator 2-norm. The residual bound uses matrix 1/infinity norms; it does not replace the theorem's operator norm. Embeddings are L2-normalized in the retrieval pipeline, though their stored norms are measured rather than assumed to be exactly one.

For machine-score decisions the experiment uses a total bound consisting of an outward-rounded real-arithmetic bound plus old/proposed float64 scoring-error envelopes. In the conditions below, use that total bound when interpreting the numerical experiment. The scorer/enclosures assume CPU IEEE-754 binary64, round-to-nearest, and gradual underflow. Boolean comparisons use exact rational arithmetic on the stored binary64 scores and bounds. The plain `per_document_bounds` utility does not itself provide these numerical safeguards.

For pre-update scores sorted by descending score:

- Global membership certificate: `m_k = s_(k) - s_(k+1) > 2 max_j B_j`.
- Pairwise membership certificate: `C = min_{i in T_k, j not in T_k} [s_i - s_j - B_i - B_j] > 0`.
- Ordered top-*k* certificate: pairwise membership certificate plus `min_{r < k}[s_(r)-s_(r+1)-B_(r)-B_(r+1)] > 0`.

The reported `min_pairwise_gap` and `min_rank_gap` are the exact `Fraction` gaps rounded to float64. Every finite float64 is an integer multiple of 2^-1074, so a nonzero exact gap cannot round to zero and the reported sign always matches the decision. The magnitude is still rounded, so read the boolean decision fields rather than recomputing a decision from reported slack. Global certification implies pairwise certification; ordered certification is the pairwise condition **and** the internal adjacent-rank condition. Pairwise certification guarantees membership only, not exact ordering, and does not imply ordered certification.

These guarantees cover only the deterministic consensus proposal. They do not cover injected post-proposal noise, model/embedding changes, different candidate construction, a biased head, multi-step compound updates without recertifying each step, or ties under an unspecified tie policy.

## Critical invariants

- Compute and persist certificate decision inputs from `scores_old`, `Theta_old`, and the deterministic `Theta_consensus - Theta_old` before assigning `Theta` or obtaining `scores_new`.
- All current certificate entry points require `noise_std=0` and reject other values, including NaN/infinity. There is no noisy certificate-run mode.
- Certificate heads must be `nn.Linear(..., bias=False)`; adding a bias changes the proof and bound.
- Use the operator 2-norm, not Frobenius/vector norms, for `W` and `DeltaW`.
- Candidate relevance/gold labels may evaluate retrieval and construct an evaluation pool, but must never enter graph construction, consensus, `DeltaW`, bounds, or certificate decisions.
- Bounds must be indexed by the same candidate/agent head that produced each score.
- Preserve strict `>` conditions. A zero margin/slack is not certified.
- Global implies pairwise; ordered implies unchanged exact ordered top-*k*; pairwise implies unchanged top-*k* membership only.
- Retrieval ranking uses descending score, then ascending candidate-slot ID, via stable sorting. Preserve this policy in certificate decisions and certificate-pipeline metrics. Graph-neighbor ties still use the shared graph function's default `np.argsort` behavior.

## Dataset conventions

`dataset_loaders.py` defaults to validation/dev data. Programmatic callers can override `dataset_slice`; the loader does not enforce a validation-only split:

- **HotpotQA**: `hotpotqa/hotpot_qa` (formerly the un-namespaced `hotpot_qa`, same commits), `distractor` configuration; contexts are joined as `title: sentences` and support titles are positive.
- **2WikiMultiHopQA**: `framolfese/2WikiMultihopQA`; assumes the Hotpot-style `context` and `supporting_facts` schema.
- **MuSiQue**: `dgslibisey/MuSiQue` dev JSONL; `paragraphs[].is_supporting` determines relevance. The raw `musique_ans_v1.0_dev.jsonl` is resolved with `hf_hub_download` at the pinned revision `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` (cache-only when HF offline mode is set) and loaded with `load_dataset("json", ...)`, because datasets 5.0.1 keys the repo-level `data_files` load differently online and offline, so its processed cache is never found offline. Manifests record repository, revision, source filename and raw-file SHA-256; the local `datasets` fingerprint is recorded separately and the old repo-level fingerprint `9eaf7aa381348ab4` is not authoritative.

Candidate construction identifies a passage by `(title, text)`. Two paragraphs sharing a title are distinct passages and both are kept; a byte-identical repeat collapses to one slot, and a repeat whose support flag disagrees raises `ValueError`. Support is read per paragraph where the source gives it (MuSiQue `is_supporting`) and by supporting title otherwise (HotpotQA, 2Wiki), so under a title-only source every paragraph beneath a supporting title counts as support. Missing supporting titles, examples with no support, and gold overflow past the candidate count all raise `ValueError` before RNG state is consumed. Construction then samples same-query distractors and pads a short pool from non-support passages of the scanned query set, one passage per padded title, excluding titles already chosen or gold for this query. Pools of 20/50 are therefore stress tests with cross-query padding, not native benchmark candidate pools. Evaluation is candidate-pool reranking, not corpus retrieval.

## Retriever conventions

- MiniLM (`sentence-transformers/all-MiniLM-L6-v2`): identity query/passage text; attention-mask mean pooling.
- E5 (`intfloat/e5-small-v2`): `query: ` and `passage: ` prefixes; attention-mask mean pooling.
- BGE (`BAAI/bge-small-en-v1.5`): query instruction `Represent this sentence for searching relevant passages: `; unmodified passages; CLS pooling.
- All pooled retrieval embeddings are L2-normalized; `proj_dim` must equal the checkpoint hidden size (currently 384 for the configured models). Inputs are truncated to 96 tokens.

Adapters directly use `AutoModel`, rather than invoking each checkpoint’s sentence-transformers implementation. Validate any changed checkpoint’s official formatting/pooling/normalization requirements before reporting results.

`adapter_metadata` compares available local checkpoint metadata without downloading it. It reports separate pooling/normalization checks, leaves formatting unvalidated, and records a different official token cap as an intentional customization. These metadata checks do not establish encoder equivalence. Unknown programmatic checkpoint names fall back to identity formatting and mean pooling; the CLI restricts checkpoint choices.

## Commands

From the repository root:

```powershell
# Main sweep entry point; do not launch before fixing the integration blocker.
# Default run kind is development, default query count is 40.
python evaluation_harness.py --pipeline full-sweep --save-per-query

# One dataset/retriever run.
python evaluation_harness.py --pipeline topk-stability --dataset hotpotqa --retriever sentence-transformers/all-MiniLM-L6-v2 --save-per-query

# Candidate-count stress test (MiniLM, 10/20/50 candidates).
python evaluation_harness.py --pipeline candidate-ablation

# Analyze only explicitly selected current-protocol metrics in fresh directories.
# Replace the example run paths before use.
python analyze_results.py --input runs/<run>/metrics.csv --out-dir runs/<new-summary-dir>
python plots.py --input runs/<run>/metrics.csv --out-dir runs/<new-plot-dir>

# Invariant suite (requires numpy, torch, scipy, networkx and imported dependencies).
python verify_invariants.py

# Complete suite, including three Hypothesis properties and the manual checks.
python -B -m pytest --hypothesis-show-statistics -q
```

`--graph-neighbors` sets `baseline_top_k`; `--top-k` is its deprecated alias. The baseline graph chooses incoming neighbors for each receiving agent, despite CLI help calling them outgoing neighbors. `--retrieval-k` independently sets retrieval/certificate depth (default 5). `--seed`, `--hops`, `--run-kind`, `--model-revision`, and `--dataset-revision` are passed to certificate pipelines. Paper mode requires nonempty revision arguments, but does not ensure they are immutable commit IDs. Full sweeps reuse one revision argument across all selected checkpoints/datasets. Several irrelevant CLI options are accepted but ignored by other pipelines; do not assume every flag applies to every pipeline.

`pyproject.toml` declares runtime dependencies and retrieval/reporting/test extras, plus pytest discovery. Versions are unpinned and no linter/type checker or CI workflow is configured. Its explicit packaging list currently omits `numerical_bounds`, `run_provenance`, `corrected_results`, and `artifact_index`; working source-tree imports do not establish a complete installable distribution.

## Results and provenance

Historical files under `results/` lack commit/package/model/dataset/seed/noise/command provenance. Their main aggregate and per-query CSVs agree on the checked certification and NDCG aggregations, but `protocol_corrected_run.log` contains different values for nominally matching configurations. Do not treat the historical summary's empirical claims as established current-protocol evidence.

Current numerical runs always create a fresh run directory with schema-3 manifests, protocol `topk-fp64-v2`, configuration, command, package/platform versions, Git state, root-Python source hashes, dataset/model metadata when available, candidate-policy text, pre-update audits, post-update scores, `metrics.csv`, `per_query.csv`, and completion hashes. CLI wrappers additionally write a schema-2 parent configuration and combined CSV exports; `--save-per-query` controls the extra combined export, not the per-run query audit. Tests also create run directories when fixtures leave `run_dir` unset.

Provenance remains incomplete: query/document identities, assembled text, frozen embeddings, graph matrices, and test/configuration source snapshots are not persisted; dataset revisions are requested values rather than independently resolved immutable revisions. Cached adapter metadata can supply a checkpoint revision even for synthetic fixtures that never ran an encoder. CSVs contain absolute manifest paths. The reader checks referenced run hashes and matching metadata, but does not bind an arbitrary input CSV's metric values to the hashed canonical CSV. Completion/inventory status is not scientific validation. Existing current-protocol artifacts are development/smoke fixtures, not a paper sweep.

Never overwrite historical `results/` or `figures/`; use a new timestamped output directory. Current certificate writers use exclusive creation.

## Testing requirements

Before committing certificate or pipeline changes:

1. Run `python -m py_compile` on all Python modules.
2. Run `python verify_invariants.py` and report all failures exactly.
3. If pytest/Hypothesis is installed, run `python -m pytest --hypothesis-show-statistics -q verify_invariants.py`; do not weaken randomized soundness checks.
4. Add tests for pre-update-only decision evaluation, deterministic ties, no-noise certification enforcement, candidate-label isolation, and any changed dataset adapter.
5. For a numerical/mathematical change, test the bound against random matrices and end-to-end deterministic consensus cases, including all three certificate implications.

## Do not do these things

- Do not use `scores_new`, post-update parameters, or gold labels to decide certification.
- Do not report noisy runs as certified/theoretical results.
- Graph neighbour ties break by ascending agent index and retrieval ties by ascending candidate slot; both use stable sorts. Do not reintroduce an unstable sort.
- Do not silently alter candidate construction, passage de-duplication, padding, formatting, pooling, normalization, or the 96-token cap.
- Do not overwrite historical result files or derived tables/figures.
- Do not replace spectral norms, relax strict inequalities, or add head bias without revising the proof, tests, and paper.
- Do not claim full-corpus retrieval, learned multi-agent specialization, or independent multi-seed uncertainty from the current code.

## Known limitations and research caveats

- The graph/consensus trajectory depends only on initialized heads and is identical across datasets and retrievers at matched alpha/hop; it is not content-, label-, or learned-agent-specialization-driven.
- The supplied historical main sweep has one fixed seed/trajectory, no independent multi-seed confidence intervals, and no run manifest. Current CLI support for a seed does not create multi-seed evidence.
- Agent identity is candidate-slot identity; heads are initialized near identity rather than learned from agent-local retrieval data.
- Candidate pools are small, relevance labels construct the evaluation pool, and padded ablations mix cross-query distractors.
- Retrieval ties now use candidate-slot order. Strict certificates decline boundary ties for membership and internal adjacent ties for ordering. Graph-neighbor tie policy and numerical platform assumptions still limit cross-environment reproducibility.
- Aggregate `bound_holds_rate` checks the maximum bound, while the stronger `per_document_bound_holds` diagnostic is available only in query rows. Summary/plot readers do not establish absence of violations.
- `numerical_bounds.py` now states its enclosure derivation and cost in its module docstring; the missing `PHASE2.md` reference is gone. Realistic-dimension performance is still unprofiled, and the passing tests are finite evidence, not a proof of all executions.
- The bound is expected to be conservative; empirical zero violations validate only executed finite cases, not the proof.

## Safe change procedure

1. State whether the change affects the proof, data protocol, retriever adapter, graph, or only reporting.
2. Inspect the primary path and update/extend invariants before implementation.
3. Keep raw inputs, pre-update certificate decisions, post-update diagnostics, configuration, versions, seed, and command in a new run directory.
4. Run the required checks and a small deterministic smoke run before an expensive sweep.
5. Regenerate analysis and plots only from the matching new raw CSVs.
6. Report exact commands, configuration, failures, changed claims, and whether the mathematical contract remains valid.
