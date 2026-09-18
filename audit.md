# Technical baseline

Inspected on 2026-09-17 at Git commit `e8ea57c52688eabf6edcecdc9c274c7afd5a7b7d` on `master`, with pre-existing uncommitted changes.

The numerical fixtures and existing tests pass. The actual dataset preparation path is incompatible with the numerical scorer's input shape, so the repository does not currently have a demonstrated working dataset-to-certificate pipeline. Historical paper results are not evidence for the current implementation.

Scope: all 20 project Python files (5,490 lines), including all tests, were read. Also inspected `AGENTS.md`, `PHASE1.md`, `pyproject.toml`, `.gitignore`, `abstract.tex`, the historical results summary, Git history/status, artifact inventories/manifests, CSV schemas and selected consistency checks, and the historical log. Installed packages and caches were treated as environment/generated files, not project source. Figures were inventoried and hashed, not regenerated or visually reviewed.

Only `AGENTS.md` and this report were edited/created for this audit. No Python source, tests, research settings, existing results, or figures were changed. No files were deleted, renamed, staged, or committed. No models/datasets were downloaded and no full paper experiments were run. Existing tests wrote three small development-run directories and refreshed ignored test caches.

## 1. Actual architecture

The repository is a collection of modules and script entry points. `evaluation_harness.py` contains most orchestration, configurations, metrics, graph construction, dataset preparation, experiment loops, CSV exports, and CLI dispatch. There is no application/service layer.

The certificate path is:

1. `parse_args` selects `topk-stability`, `full-sweep`, or `candidate-ablation`. The corresponding `main_*` function assembles `TopKStabilityConfig`, validates noise and retrieval depth, creates a fresh parent output directory, and writes a schema-2 configuration. Default configuration is 40 queries, 10 candidate slots, 384 dimensions, five hops, retrieval depth 5, three graph neighbors, seed 42, alpha 0.6, and zero noise. The default full sweep varies three datasets, three retrievers, and five alphas. These loops were inspected, not executed.
2. `_prepare_hotpotqa_run(..., dataset=...)` calls `load_dataset_records`. HotpotQA and 2Wiki use the same assumed context/support-title schema. MuSiQue converts paragraph support flags into a set of gold titles. Default splits are validation/dev. Programmatic split overrides are unrestricted.
3. Candidate assembly keeps the first paragraph per title, includes available gold titles, samples same-query negatives, and pads from non-positive passages of scanned examples. It shuffles slots with a seeded generator. Overflow of available unique gold titles raises an exception. Incomplete candidate pools and examples with no remaining positive label are skipped. Missing gold titles and duplicate-title loss are not validated.
4. Frozen `AutoTokenizer`/`AutoModel` embeddings use checkpoint-specific prefixes and pooling, a 96-token cap, and L2 normalization. MiniLM/E5 use masked mean pooling; BGE uses CLS. The hidden dimension must match `proj_dim`. `_build_agent_shards` assigns one passage slot to each agent and stores query/document tensors shaped `(1, dimension)` plus a relevance tensor. Heads are seeded near-identity matrices, initially float32, with no bias in certificate runs.
5. `run_topk_stability_experiment` starts a schema-3 per-run manifest and copies heads into CPU float64. At each hop it constructs a cosine graph from flattened head parameters. For each receiver `v`, it chooses similar agents `u` and adds `u -> v`. Incoming edge weights are normalized and used in `P.T @ Theta`; alpha blends each receiving head with that incoming average. Isolated receivers keep their old parameters. Graph connectivity is a diagnostic, not a certificate decision input.
6. Pre-update scores use `bilinear_score`. This is the current execution blocker: the scorer indexes one-dimensional query/document vectors, while step 4 supplies row matrices. A normal nontrivial embedding dimension fails before the first certificate. The remaining sequence works in the existing one-dimensional numerical fixtures.
7. The deterministic proposal is formed without assigning heads. `spectral_norm_upper`, `update_norm_upper`, and `certified_bounds` construct real-arithmetic upper bounds and old/proposed score-roundoff envelopes. `full_certificate` takes old scores and total bounds and returns immutable decisions. Old/proposed heads, old scores, bounds, embedding norms, and decision JSON are written before mutation. The proposed head is used to bound roundoff by magnitudes; its scores are not evaluated to decide certification.
8. Heads are assigned exactly the stored float64 proposal, without noise. New scores enter `validate_certificate`, membership/order-change diagnostics, and retrieval metrics. Labels enter retrieval metrics only at this stage; shard `y` is unused by the certificate path. Stable candidate-slot order is imposed before invoking shared legacy metric functions.
9. Each run always writes `metrics.csv`, `per_query.csv`, and completion hashes. CLI wrappers can also write combined sweep/comparison/summary CSVs. `--save-per-query` controls extra combined exports, not the per-run audit. `corrected_results.py` reads selected CSVs and checks referenced metadata/hashes. `analyze_results.py` and `plots.py` produce fresh descriptive summaries/plots, not the historical paper tables/figures.

The other paths are preserved legacy work: synthetic linear agents and HotpotQA with DistilGPT2 embeddings, a random projection, biased heads, and cosine or Fisher/SIR routing. They use `agent_retrieval_graph.py`, `fisher_information_geometry.py`, `theorem1_diagnostics.py`, and `margin_diagnostics.py`. In those paths labels can enter Fisher calculations. That behavior must not be attributed to the certificate experiment. Their writers still default to overwriting `results/` and `figures/`; those entry points were not run.

AST comparison with HEAD found 29 shared top-level harness functions/classes: 28 unchanged; `main_hotpotqa` changed through extraction of shared preparation. The four standalone legacy modules have no Git diff. This does not constitute a live legacy dataset regression run.

## 2. Git state

- Branch: `master`; HEAD is `e8ea57c`, dated 2026-08-18. Its parent is `97f7c19`. No remotes were listed. Nothing is staged.
- Modified tracked source/document files: `evaluation_harness.py`, `verify_invariants.py`, `abstract.tex`.
- Modified tracked generated files: `__pycache__/evaluation_harness.cpython-314.pyc`, `__pycache__/verify_invariants.cpython-314.pyc`, and `__pycache__/verify_invariants.cpython-314-pytest-9.0.3.pyc`. Seven bytecode files are tracked overall; ignore rules do not untrack them.
- Before this audit's checks: 6 modified tracked files and 136 untracked files. After the existing suite: 167 untracked files, including 119 under `runs/`. This report adds one more, for 168. `AGENTS.md` was already untracked, so its correction does not change that count.
- The final untracked categories are 16 root files, 4 test files, 12 figures, 17 result/table files, and 119 run files. `.venv/`, `.hypothesis/`, `.pytest_cache/`, and untracked bytecode are ignored.
- HEAD alone is not the current certificate implementation: essential imported modules are untracked. A clean checkout cannot reconstruct this working tree.
- Git reported LF-to-CRLF normalization notices for existing tracked text changes. One status invocation warned that the global ignore file could not be read; repository-local state remained readable. No Git configuration was changed.

The test-created run directories are:

```text
runs/20260917T201604_35b55fd5f0964aa68b60fe283b5c8f72/
runs/20260917T201605_ffd492e5b3214d4c8632992a9488551b/
runs/20260917T201606_62fabf8a718e4f3bb17da5d8789bb356/
```

## 3. Files to commit eventually

After reviewing the defects below, the source baseline should include the modified harness and invariant suite, plus all currently untracked implementation modules:

```text
topk_stability.py
numerical_bounds.py
dataset_loaders.py
model_adapters.py
run_provenance.py
corrected_results.py
artifact_index.py
analyze_results.py
plots.py
phase1_smoke.py
tests/conftest.py
tests/test_phase1.py
tests/test_certificate_properties.py
tests/test_numerical_bounds.py
pyproject.toml
.gitignore
AGENTS.md
PHASE1.md
audit.md
```

Retain `PHASE1.md` as historical notes, not current implementation documentation. `abstract.tex` needs a separate claim/mathematics review before committing it as an accurate paper abstract. Do not bundle historical CSVs, log files, figures, or generated test runs into a source commit without an explicit artifact-retention/provenance decision. Preserve them meanwhile. No commit was made by this audit.

## 4. Stale, redundant, and missing files

- Tracked `.pyc` files are generated and interpreter-specific; they should eventually leave version control. No deletion/untracking was performed.
- `.venv/`, Hypothesis/pytest caches, and untracked bytecode are local environment artifacts, not source deliverables.
- `PHASE1.md` contains useful historical checks but describes superseded float32 scoring, missing error envelopes, optional persistence, old readers, and absent Git metadata.
- `numerical_bounds.py:6` references `PHASE2.md`, which does not exist. No replacement derivation file was invented during this audit.
- Historical `results/`, `results/tables/`, `figures/`, `protocol_corrected_run.log`, and old smoke directories are retained evidence, not automatically unnecessary files. Their claims/provenance need review before reuse.
- Per-run `configuration.json` duplicates the schema-3 manifest, while CLI parent configurations still use schema 2. Parent aggregate/query exports duplicate per-run data but are not included in per-run completion hashes.
- Some tests create persistent run folders outside `tmp_path`; these accumulate fixtures labeled `development`, often with default HotpotQA/MiniLM metadata despite using handcrafted two-dimensional vectors.
- `pyproject.toml` omits four present, required modules from its explicit package list: `numerical_bounds`, `run_provenance`, `corrected_results`, and `artifact_index`. A source-tree test pass does not verify an installed wheel. No package build/install was attempted.
- There is no CI workflow, dependency lock, configured linter/type checker, or standalone project README. The missing README is a documentation gap, not a broken project-local runtime reference. The README and JSON names in `model_adapters.py` refer to optional external model-cache files, not missing project files.
- Thin reporting wrappers, unused imports, repeated diagnostic evaluation, stale two-word comments, and section banners remain. None warranted a rename or structural cleanup during this baseline.

At audit completion, the inventory found 39 historical result/figure files, nine manifest-bearing development runs, one current-protocol smoke run, and two older configuration-only run records. All ten current-protocol runs passed their stored completion hashes. That is file integrity only; none is a new paper sweep.

## 5. Verification results

Interpreter: `.venv/Scripts/python.exe`, Python 3.14.2, Windows AMD64. Observed packages: NumPy 2.5.3, PyTorch 2.14.0, SciPy 1.18.1, NetworkX 3.6.1, Matplotlib 3.11.2, pandas 3.0.5, pytest 9.1.1, Hypothesis 6.168.0, datasets 5.0.1, transformers 5.17.0. These are observed versions, not validated compatibility ranges.

| Check | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -B -m pytest --hypothesis-show-statistics -q` | 88 passed, no skips/failures, 20.38 seconds |
| Three Hypothesis properties | 100 passing, 0 failing, 0 invalid cases each; 300 generated cases total |
| `.\.venv\Scripts\python.exe -B verify_invariants.py` | 58/58 checks passed; no failures |
| `.\.venv\Scripts\python.exe -B -m pytest --hypothesis-show-statistics -q verify_invariants.py` | 58 passed, 6.18 seconds; this file has no Hypothesis-decorated properties |
| Compilation of all 16 root modules and 4 test modules | Passed |
| `git diff --check` and whitespace checks of the two audit documents | Passed; Git emitted existing LF-to-CRLF notices |
| SHA-256 comparison before/after documentation edits | All 20 Python source/test files and 39 historical result/figure files unchanged |

Compilation command:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'topk-baseline-pycache'
$modules = Get-ChildItem -File -Filter *.py
$testModules = Get-ChildItem tests -File -Filter *.py
.\.venv\Scripts\python.exe -m py_compile @($modules.FullName) @($testModules.FullName)
```

No tests, tolerances, assertions, or trial counts were changed. The full-suite command also satisfies the requested Hypothesis run. The manual checks are included in the 88-test pytest total, not 58 additional distinct tests. No separate smoke script or full dataset experiment was run in this audit.

Coverage gaps: no test traverses real preparation into the numerical runner; no dataset source-schema fixtures; no reporting/provenance-reader integrity regression tests; no install-from-wheel check. The first two Hypothesis properties exercise the plain real-arithmetic utilities, and the third checks numerical score/norm envelopes in three dimensions. The nine scale/step numerical cases use two-dimensional diagonal heads. These do not exhaustively validate full-dimensional numerical behavior. `tests/conftest.py` imports Hypothesis unconditionally, so the test extra is required despite an `importorskip` in one test module.

## 6. Code and mathematics

| Requirement | Assessment |
| --- | --- |
| Pre-update-only certificate | Satisfied in the numerical runner/API: frozen decisions and pre-update persistence precede mutation; no new scores are accepted by `full_certificate`. Dataset integration is blocked separately. |
| New scores are diagnostic only | Satisfied for certification; they also evaluate post-update retrieval quality. The older empirical certificate API is explicitly separate. |
| Zero noise | Enforced by certificate entry points and decision API, including rejection of NaN/infinity. Legacy noise behavior is outside this guarantee. |
| Deterministic retrieval ties | Stable descending scores, ascending candidate-slot ID, including certificate-pipeline metrics. Graph similarity ties retain default unstable sorting and are a distinct reproducibility issue. |
| Zero relevant margin/slack cannot certify | Exact comparison uses strict positivity. Boundary ties decline membership and ordered certificates; internal ties decline ordered certification. Reported rounded slack can nevertheless be zero for a positive exact slack. |
| Global implies pairwise | Correct in the implementation for nonnegative valid bounds; covered by existing tests. |
| Pairwise guarantees membership | Correct conditional on the per-candidate bounds covering the actual score changes. Labels and new scores do not decide it. |
| Ordered guarantees order | Correct: pairwise membership AND all internal adjacent-rank comparisons. Pairwise alone does not imply ordered. |
| Operator 2-norm | Plain utility uses `ord=2`; experiment uses an upper enclosure of that same norm. Matrix 1/infinity norms bound only numerical residuals. |
| Float64 safeguards | Double heads, explicit CPU sequential score operations, outward-rounded real bounds/error envelopes, and exact rational decisions are present. Frozen embeddings and initialization originate in float32; they are treated as fixed stored inputs. This is not certification of the encoder computation or a machine-checked proof. |
| Independent graph/retrieval settings | `baseline_top_k` and `k` are distinct and mapped independently by CLI. Graph help incorrectly calls the chosen incoming neighbors outgoing. |
| Gold-label isolation | Satisfied after candidate construction in the certificate path. `relevance` evaluates metrics; shard `y` is unused. Candidate pools themselves intentionally depend on labels. Bounds assume the same query vector across slots, as preparation supplies; direct callers are not checked for this condition. |
| No silent gold loss | Only available unique-title overflow is protected. Missing source gold and duplicate-title supporting text can be silently omitted; the unconditional requirement is not met. |

Reproduced defects and inconsistencies:

1. **Embedding shape:** `_build_agent_shards` (`evaluation_harness.py:683`) yields `(1, 2)` vectors even in a two-dimensional miniature of the real preparation path. Passing them unchanged to `bilinear_score` produces `IndexError: index 1 is out of bounds for axis 0 with size 1`. Passing the same shapes to `certified_bounds` produces `TypeError: only 0-dimensional arrays can be converted to Python scalars`. Existing tests hand-construct `(2,)` vectors instead.
2. **Gold retention:** `RawExample('missing', {'A','B'}, [('A','gold a'),('C','distractor')])` produces an accepted two-candidate pool with only A relevant; no missing-B error occurs. With two A paragraphs, first-occurrence de-duplication retains the first text and drops the later supporting text. For MuSiQue, converting paragraph support flags to titles can also assign that support label to the wrong same-title paragraph. No claim is made here about how often actual datasets contain this condition.
3. **Rounded slack differs from the decision:** `full_certificate(np.array([1., -2.**-54]), 1, np.array([.5,.5]))` reports `min_pairwise_gap == 0.0` with `pairwise_certified == True`. Its exact slack is positive (`2**-54`), so the decision is mathematically correct; the persisted/displayed scalar is not an exact representation of the decision criterion. The existing random slack-sign check misses this edge.
4. **Different bound diagnostics:** with old scores `[2,0]`, bounds `[0.1,10]`, and new scores `[2.2,0]`, `bound_holds` is true, `per_document_bound_holds` is false, and aggregate `bound_holds_rate` is 1.0. This is expected for the weaker maximum-bound check, but that aggregate cannot validate the candidate-specific bounds needed by the pairwise certificate.
5. **CSV integrity boundary:** copying a valid run CSV to a temporary location and changing `pairwise_certificate_rate` to `123.0` still passes `load_corrected`. Original canonical run files and hashes remain untouched. The reader verifies those referenced originals and configuration, not the supplied CSV's metric values or ranges. It also does not require expected hop coverage, all expected completion artifacts, or a nonempty completion hash mapping. The inventory rejects an empty mapping, so the two readers have different checks.
6. **Paper implication:** `abstract.tex:17` states that each certificate is implied by the preceding one. Counterexample: scores `[3,3,1]`, zero bounds, and `k=2` produce global=true, pairwise=true, ordered=false. Only global -> pairwise and ordered -> pairwise hold generally. The abstract also writes a single shared W while the experiment uses candidate-specific heads, and omits the numerical envelope distinction.

The historical main CSV has 225 aggregate rows and 222,525 query rows; the candidate ablation has 45 hop rows. Regrouping query rows by dataset/retriever/alpha/hop matches aggregate pairwise rate, ordered rate, and NDCG in all 225 cells within `1e-10`. This is internal consistency, not run provenance. For HotpotQA/MiniLM, alpha 0.1, hop 1, the aggregate CSV reports NDCG 0.7373625511 and bound 4.5457377055, while the log reports NDCG 0.7738 and bound 0.2939. The log stops while loading MuSiQue/E5 and does not establish completion of a matching full sweep.

## 7. Research-validity risks

Severity describes the consequence of relying on the current repository for paper claims, not a claim that every issue has already corrupted an experiment.

| Severity | Risk and evidence |
| --- | --- |
| CRITICAL | Publishing the current abstract/results summary as validated evidence: the implication chain is false, historical provenance is unresolved, and the newer fixtures cannot validate the old empirical claims. |
| HIGH | Real preparation cannot enter the numerical certificate path because of embedding shapes. Passing fixture tests currently conceal this integration failure. |
| HIGH | Gold preservation is only title-based; missing gold and duplicate titles can change the evaluation pool/relevance labels without an explicit failure. Source-schema adapters are not tested end to end. |
| HIGH | The current result reader can accept altered metric values by referencing a valid original run. Summaries omit violation counts and cannot serve as a soundness gate. |
| HIGH | Reproduction is incomplete: no frozen embeddings, assembled records/document identities, or graph matrices; dataset revision strings are not resolved immutable commits; paper mode accepts movable revision names; one revision argument is shared across all repositories in a sweep. |
| HIGH | Numerical assurance is stronger than the old float32 path but lacks the referenced derivation and realistic integration coverage. CPU rounding/underflow assumptions are declared rather than enforced. Small randomized/exact-arithmetic probes are not proof of all executions. |
| MEDIUM | Rounded slack can disagree with exact decision signs, and aggregate bound diagnostics test only the maximum bound. Analyses reconstructing decisions or per-document soundness from these summaries can be misleading. |
| MEDIUM | Historical rule-of-three limits and correlation p-values treat repeated query/hop/configuration observations as stronger statistical evidence than their dependence warrants. Summed counts across certificate types also overlap. No independent multi-seed uncertainty is established. |
| MEDIUM | Agent identity is candidate-slot identity, heads are initialized rather than learned, and graph trajectories depend on heads rather than query content. Candidate-count changes alter both pools and graph/head population. Padded pools are reranking stress tests, not corpus retrieval or end-to-end QA. |
| MEDIUM | Adapter evidence is local metadata comparison, not encoder equivalence. Cached metadata can label synthetic fixtures as using a real model revision. Unknown programmatic models silently use fallback formatting/pooling. |
| MEDIUM | Dependency versions are unpinned; packaging omits four modules; source-tree tests mask installation problems. No CI or clean-install baseline exists. |
| MEDIUM | Configuration checks cover noise, retrieval depth, and presence of paper revision strings, but not all counts, alpha finiteness, shared-query assumptions, or graph-neighbor range. Graph ties have no explicit stable policy. Several CLI flags are silently irrelevant to some pipelines. |
| LOW | Generated bytecode is tracked, tests accumulate run artifacts, metadata/configuration exports are duplicated, and comments/names/documentation have drifted. Absolute manifest paths also make artifacts difficult to relocate. |

No counterexample to the central real-arithmetic bilinear bound or the exact global/pairwise/ordered decision logic was found under their stated input assumptions. That conclusion is limited by the integration and coverage gaps above.

## 8. Documentation inconsistencies

`AGENTS.md` was corrected where stale: API/control flow, zero-noise enforcement, ties, overflow behavior, float64 numerical path, independent CLI settings, dependency/test configuration, reporting inputs, manifests, and limitations. It now records the shape blocker and distinguishes implemented safeguards from unverified end-to-end behavior.

Other documents were left unchanged:

- `PHASE1.md` is a historical record and does not describe the latest scoring, diagnostics, provenance, or reporting behavior.
- `abstract.tex` contains the incorrect implication chain, empirical numbers without current-run provenance, a shared-head formula without the candidate-specific experimental qualification, and a noisy-setting statement with no current certificate-run mode supporting it.
- `results/PAPER_RESULTS_SUMMARY.md` labels empirical claims as supported/safe despite missing provenance and dependence between observations. Current analysis scripts no longer regenerate that document or its tables.
- CLI help says outgoing graph neighbors even though edges select incoming neighbors per receiver. Sample-count help mentions only the legacy pipeline; alpha-sweep help omits its full-sweep use. Extra per-run query persistence is not explained by `--save-per-query` help.
- Source comments such as `all gold kept`, the log message `paper-ready summary table`, and the missing numerical derivation reference overstate what the implementation establishes.
- Parent configurations use schema 2 and null top-level revision placeholders; child manifests use schema 3. The artifact index labels any configuration-only run as an old smoke run, although CLI parent directories can also meet that condition.

## 9. Recommended Phase 2 work

These are recommendations only; none was implemented in this audit.

1. Add a regression using the actual shard builder and repair the row-vector/one-dimensional-vector boundary without altering embeddings or the legacy Fisher path. Verify preparation, scoring, all decisions, and serialization together using an offline encoder/dataset fixture.
2. Make gold-retention failures explicit, including missing gold and duplicate-title ambiguity, and add source-schema fixtures for all three datasets. Any change to candidate identity or selection requires a documented protocol decision rather than silent cleanup.
3. Bind supplied result rows to canonical audited data; test altered CSVs, incomplete hashes, missing hops, aggregate/query mixtures, relocatable manifests, and metadata mismatches. Keep per-document failures and certificate violations visible in reporting.
4. Document and review the numerical enclosure derivation and assumptions. Address exact-decision versus rounded-slack reporting, and exercise non-diagonal consensus and realistic dimensions before claiming numerical coverage. Profile the reference implementation before scheduling expensive runs.
5. Align run metadata with actual inputs: resolved immutable revisions per repository, true fixture labels, realized query/candidate identities, and enough stored inputs for replay. Consolidate parent/child configuration semantics only after the current formats are understood.
6. Complete packaging and validate an offline clean installation. Move test outputs into temporary directories, then define artifact retention and remove generated bytecode from tracking only in an explicitly reviewed cleanup. Establish repeatable environment/CI checks without weakening current tests.
7. Correct the abstract and distinguish historical empirical claims from current implementation claims. Revise dependent-observation uncertainty statements. Preserve historical results until an explicit archival decision.
8. Only after these gates pass, plan a separately authorized, versioned experiment run with matching raw inputs and derived outputs. Do not treat this audit or its green tests as authorization to launch the full paper experiments.
