# Phase 1 corrections

AGENTS.md remains the repository guide. This document records the Phase 1 implementation
changes to its descriptions of the old certificate API, ties, gold overflow, and CLI.
The spectral bound, strict inequalities, candidate/head indexing, retriever adapters,
and preserved Fisher/SIR implementation are unchanged.

## API and update sequence

```python
decision = full_certificate(scores_old, k, per_document_bounds, noise_std=0.0)
# Apply the deterministic proposal only after storing every query's decision.
diagnostics = validate_certificate(decision, scores_new)
```

`CertificateDecision` is frozen and stores tuple copies of scores, bounds, candidate
IDs, ordered selections, margins, and all decisions. No predictive decision API
accepts post-update scores. `predictive_certificate` and `dual_certificate` use the
same pre-update signature, with a scalar bound and per-document bounds respectively.
Aggregate predictive/dual/full diagnostics now take `(decisions, scores_new)`.
These are intentional Python API changes. The older empirical
`topk_stability_certificate` remains a post-hoc diagnostic, outside this pipeline.

Scoring reads current heads without writing them. Each hop computes the proposal,
bounds, and every immutable decision before applying any head update. Validation
reads the stored decisions; it cannot revise them. Nonzero, negative, NaN, and
infinite `noise_std` values fail before loading data/models in certificate entry
points, and before constructing heads in the numerical runner. The low-level
decision API also rejects them. Legacy Fisher/SIR noise behavior is unchanged.

## Ranking and candidates

Ranking is descending score, then ascending zero-based candidate-slot ID, using
stable sorting. IDs are the positions in the assembled pool and stay fixed across
hops. They are deterministic for a fixed dataset order, pool construction, and
seed, not persistent document IDs across independently constructed pools.
Certificate comparisons and certificate-pipeline retrieval metrics use this same
policy. Shared legacy metric and graph functions remain unchanged.

A boundary tie rejects membership and ordered certificates. An internal top-k
tie rejects the ordered certificate but can still allow membership certification.
All decisions retain strict `>` comparisons.

Gold overflow raises `ValueError` with the question, gold count and candidate slot
count, before consuming RNG state. Valid examples keep the existing de-duplication,
sampling, shuffling and padding behavior. This does not repair missing source gold
paragraphs or establish that padded pools are native benchmark candidate pools.

## CLI and output compatibility

Use `--graph-neighbors` for routing and `--retrieval-k` for retrieval/certification
depth (default 5). `--top-k` retains its graph meaning with a visible deprecation
warning. Supplying both graph spellings is an error. Retrieval depth is accepted
only by certificate pipelines. Candidate ablation now honors `--candidate-count`.

Certificate entry points create a fresh timestamped subdirectory under `runs/`,
or under the parent supplied by `--out-dir`. Certificate CSV writers refuse to
overwrite existing files. Existing `results/`, figures and `abstract.tex` are untouched.

Schema 2 uses `ndcg_at_k`, `f1_at_k`, and `recall_at_k` columns, with explicit
`retrieval_k`, `graph_neighbors`, and `noise_std` on every metric row. Old analysis
and plotting scripts still expect `*_at_5`; adapting them is deferred. Do not feed
schema 2 CSVs into those scripts or mix them with historical CSVs.

Every CLI numerical run writes configuration, command, Python/package versions,
seed and ranking policy. Each combination has its own directory with pre-update
parameter/proposal arrays, scores, bounds, embedding norms and decision JSON saved
before mutation; post-update score arrays are separate. Direct numerical callers
can set `cfg.run_dir` to persist this audit; immutable decisions are always stored
in memory. Query-level metric CSVs remain optional. Model/dataset revision fields
are explicitly unresolved (`null`), so this is not a fully pinned reproduction
manifest. Frozen embeddings, dataset records and repository commit/dirty state
are not captured; rerunning encoders requires additional provenance work.

Old noiseless, untied, valid-pool calculations retain the same formulas. Old noisy
certificate claims are unsupported; ties now have defined empirical outcomes;
gold-overflow examples now fail rather than produce truncated gold pools.
Historical files cannot be reclassified as corrected evidence without rerunning.

## Dependencies and verification

`pyproject.toml` declares only imported runtime packages: NumPy, PyTorch, SciPy,
NetworkX and Matplotlib; retrieval extras are datasets/transformers, reporting
adds pandas, and testing adds pytest/Hypothesis. Package compatibility ranges are
unresolved and intentionally unpinned: the repository did not supply a validated
version range or lockfile. Python 3.10+ is required by strict zip usage. With explicit
user approval, `python -m pip install hypothesis` installed Hypothesis 6.168.0 and
its dependency sortedcontainers 2.4.0 into the existing user-site environment.
These observed versions are not asserted compatibility ranges. No formatter or
static checker is configured. Pytest now discovers `verify_invariants.py` as well
as `tests/`.

Verification commands (PowerShell, repository root):

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'topk-phase1-pycache'
$modules = Get-ChildItem -File -Filter *.py
$tests = Get-ChildItem tests -File -Filter *.py
python -m py_compile @($modules.FullName) @($tests.FullName)
python -B -m pytest -q
python -B verify_invariants.py
python -B -m pytest --hypothesis-show-statistics -q verify_invariants.py
python -B phase1_smoke.py
git diff --check -- evaluation_harness.py verify_invariants.py
```

Runtime verification needs access to the already installed user-site packages
outside this environment's filesystem sandbox. No datasets or models are loaded
by the tests or smoke script. Hypothesis was initially missing; its installation
was subsequently authorized by the user so the property tests could execute.

Regression tests cover immutable snapshots and changed post-update scores,
multi-query ordering before mutation, persistence before mutation, boundary and
internal ties, label isolation, gold overflow/RNG preservation, noise rejection,
CLI separation and output fields. Two Hypothesis properties cover arbitrary random
matrix updates and all three implications under deterministic consensus. Existing
randomized checks retain their trial counts and assertions.

## Remaining mathematical and methodological limits

The proof describes real arithmetic. Float32 scoring and parameter operations can
exceed a saturated bound by more than the existing `1e-9` empirical tolerance.
The first smoke fixture, with collinear vectors, failed
`assert row["per_document_bound_holds"]` at this numerical edge (no ranking change).
The final smoke uses normalized non-collinear documents, with slack in the bound.
Neither the mathematical formula nor diagnostic tolerance was relaxed. A certified
floating-point error envelope remains future work; this implementation does not
claim machine-verified arithmetic soundness.

The first pytest run also exposed a stale `score_all(Theta)` call after scoring was
made read-only (`TypeError: ...score_all() takes 0 positional arguments but 1 was
given`); it was corrected to `score_all()` and the suite rerun.

Other limitations remain: fixed seed/head trajectory, candidate-slot agents,
initialized rather than learned heads, small label-constructed reranking pools,
cross-query padding, and unpinned model/dataset revisions. No end-to-end QA,
full-corpus retrieval, multi-seed uncertainty, or new paper result is established.

## Verification results for this change

- Compilation of all root and test Python modules: passed; bytecode redirected to
  the temporary `topk-phase1-pycache` directory.
- Before Hypothesis installation, `python -B -m pytest -q`: **74 passed, 1 skipped**,
  12.02 seconds. The skipped module contains the two Hypothesis properties. The initial run was
  **1 failed, 67 passed, 1 skipped** before fixing the stale scoring call.
- After the authorized installation, `python -B -m pytest --hypothesis-show-statistics -q`:
  **76 passed, no skips**, 102.88 seconds. Each of the two Hypothesis properties
  generated **100 passing, 0 failing, 0 invalid cases** (200 generated cases total).
  No test assertions, trial counts, bounds or tolerances were changed for this rerun.
- `python -B verify_invariants.py`: **58/58 passed**, no failures.
- `python -B -m pytest --hypothesis-show-statistics -q verify_invariants.py`:
  **58 passed**, 5.28 seconds after Hypothesis installation. The earlier attempt
  failed before collection with `unrecognized arguments: --hypothesis-show-statistics`
  because Hypothesis had not yet been installed.
- `python -B phase1_smoke.py`: final normalized fixture passed two hops, all three
  certificates, per-document bounds, and zero violations. Output:
  `runs/20260915T194919_ylgfu389/`. The initial failed collinear fixture's audit is
  retained separately in `runs/20260915T194841_zwjhyrct/`.
- `git diff --check -- evaluation_harness.py verify_invariants.py`: passed.
  TOML parsing also passed. No formatter/linter/type checker is configured.
- Compared 22 shared/legacy function and class bodies against the pre-edit
  workspace snapshot: unchanged, including both legacy runners, their preparation,
  retrieval metrics, graph construction and Fisher/SIR configuration. Standalone
  preserved graph/Fisher/diagnostic files have no Git diff. Their existing invariant
  checks pass; no expensive legacy dataset experiment was run.

The installed environment emitted `RequestsDependencyWarning` about its
urllib3/chardet/charset_normalizer versions. It did not prevent runtime checks.
The sandbox's default Python could not see NumPy; runtime checks were performed
with approved access to existing user-site packages. Only Hypothesis and its
sortedcontainers dependency were installed, following explicit user approval.

Intentional source/configuration changes: `topk_stability.py`,
`evaluation_harness.py`, `dataset_loaders.py`, `verify_invariants.py`,
`pyproject.toml`, `tests/test_phase1.py`, `tests/test_certificate_properties.py`,
`phase1_smoke.py`, and this document. New numerical artifacts are confined to the
two run directories above. Test execution also refreshed Python/pytest caches.
Pre-existing workspace changes, including `abstract.tex`, were retained.
