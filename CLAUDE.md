# CLAUDE.md

## Project

Certifying top-k retrieval stability under multi-agent communication.

Agents each hold a bilinear retrieval head. A consensus step blends each agent's
head with its incoming neighbours' heads. The research question is whether a
certificate computed *before* that step can guarantee the query's top-k result is
unchanged *after* it. The certificate must depend only on pre-update information:
the current head, the proposed update, and the embedding norms.

## Core mathematical model

Retrieval score for a bias-free head `W`:

    s_W(q, d) = (Wq)^T (Wd)

Communication proposes:

    W' = W + DeltaW

Per-candidate score-change bound, from submultiplicativity of the operator
2-norm:

    B(q, d) = ||q||_2 ||d||_2 [ 2 ||W||_2 ||DeltaW||_2 + ||DeltaW||_2^2 ]

Let `T_k` be the current top-k set. The **pairwise membership certificate** holds
when, for every `i` in `T_k` and every `j` outside `T_k`:

    s_i - s_j > B_i + B_j

This certifies the top-k *set* is unchanged.

The **global certificate** is the weaker sufficient condition `m_k > 2 B_max`,
where `m_k` is the boundary margin. Global implies pairwise.

**Ordered certification** requires the pairwise membership condition **and** the
internal adjacent-rank conditions

    s_(r) - s_(r+1) > B_(r) + B_(r+1)   for r = 1, ..., k-1

Pairwise certification alone does **not** imply ordered certification. A tie
inside the top-k (for example scores `[3, 3, 1]`, zero bounds, `k = 2`) is
certified for membership and rejected for ordering. Do not write or repeat the
claim that each certificate implies the next; only `global => pairwise` and
`ordered => pairwise` hold.

## Non-negotiable invariants

- Certification uses pre-update information only. `full_certificate` accepts old
  scores and bounds, and has no parameter for post-update scores.
- Post-update scores are diagnostic only. They enter `validate_certificate`,
  which never recomputes a decision.
- Certificate-validation experiments require `noise_std = 0`. Noise is rejected
  before any dataset load, model load, or head mutation.
- Heads used by the certificate model are bias-free (`nn.Linear(..., bias=False)`).
  The legacy Fisher/SIR path uses biased heads; that is a different model.
- The operator 2-norm is used. `spectral_norm_upper` returns a validated upward
  enclosure of it. Matrix 1/infinity norms appear only to bound numerical
  residuals, never as the score bound itself.
- Certificate comparisons use unrounded values. Decisions are taken in exact
  `Fraction` arithmetic on the binary inputs; floats are produced only afterwards
  for reporting. Because every finite float64 is an integer multiple of 2^-1074,
  a nonzero exact slack cannot be reported as zero, so `min_pairwise_gap` and
  `min_rank_gap` always carry the decision's sign. Their magnitude is rounded, so
  read the boolean decision fields rather than recomputing a decision from slack.
- Ties and zero margins cannot certify. All comparisons are strict inequalities.
- NaN/Inf cannot certify. Non-finite scores, bounds, or enclosures raise.
- Gold labels cannot influence routing or certification. `relevance` is consumed
  only by post-update retrieval metrics; the graph depends on head parameters.
- Supporting/gold passages must never be silently dropped. A passage is
  identified by `(title, text)`, so a repeated title keeps both paragraphs.
  Missing supporting titles, an example with no support, a byte-identical repeat
  whose support flag disagrees, and gold overflow all raise `ValueError` before
  the RNG is consumed.
- Graph neighbour count and retrieval k are separate settings:
  `baseline_top_k` / `--graph-neighbors` versus `k` / `--retrieval-k`.
- Ordering is deterministic and seed-independent. Retrieval ties break by
  descending score then ascending candidate-slot ID; graph neighbour ties break
  by ascending agent index. Both use stable sorts.
- Result artifacts are tied to run metadata/provenance. A CSV is readable as
  evidence only through its `.audit.json` sidecar and the hashed canonical run.
- A `paper` run requires reproducible provenance: explicit `model_revision` and
  `dataset_revision`, a clean working tree at a known commit, and an encoder that
  actually loaded (`resolved_model_revision`). `development` and `smoke` runs may
  be dirty, and their manifests record that they were. Because there is one
  revision slot per kind, a `paper` sweep must pin one dataset and one retriever
  per invocation.
- Historical runs are not paper evidence. `artifact_index.py` reports
  `paper_claims_allowed: false` for every artifact, labels a verified run that
  never loaded an encoder `development_fixture`, and leaves `results/`/`figures/`
  as `legacy_unverified`.
- Nothing overwrites an existing output. Certificate runs open every artifact
  with mode `x`; the historical Fisher/SIR writers now do too and default to a
  fresh `runs/legacy/<pipeline>_<timestamp>_*` directory instead of `results/`
  and `figures/`.
- A run directory is relocatable but keeps its name. Exports cite their manifest
  relative to themselves, and `validate_run` still requires the directory name to
  equal the `run_id`.
- Every certificate slot shares one query embedding, and the run rejects shards
  that do not. The per-candidate bound assumes a single `||q||` per query.
- Degenerate settings are rejected before a run directory exists: non-finite
  `diffusion_alpha`, `n_hops < 1`, `n_eval_queries < 1`, negative
  `baseline_top_k`, and `k` outside `0 < k < n_agents`.
- Old Fisher/SIR code (`fisher_information_geometry.py`,
  `theorem1_diagnostics.py`, `margin_diagnostics.py`, `agent_retrieval_graph.py`,
  and the synthetic / `main_hotpotqa` paths in the harness) is historical. Do not
  casually modify it.

## Repository map

| File | Role |
| --- | --- |
| `evaluation_harness.py` | Orchestration: configs, dataset preparation, embedding, graph construction, the certificate experiment loop, CSV writers, CLI. Also holds the historical synthetic and Fisher/SIR pipelines. |
| `topk_stability.py` | Certificate core: `full_certificate`, `validate_certificate`, `CertificateDecision`, bound helpers, aggregation. |
| `numerical_bounds.py` | Outward-rounded float64 enclosures: `spectral_norm_upper`, `update_norm_upper`, `bilinear_score`, `score_roundoff_bound`, `certified_bounds`. |
| `dataset_loaders.py` | The three datasets, `RawExample`, and candidate-pool assembly with the gold-preservation rules. |
| `model_adapters.py` | Per-retriever query/passage formatting, pooling, and cached-metadata inspection. |
| `run_provenance.py` | Run manifests, artifact hashing, git state, `validate_run`, CSV audit sidecars. |
| `corrected_results.py` | Strict result reader: binds CSV rows back to hashed canonical run artifacts. |
| `artifact_index.py` | Read-only inventory marking which artifacts are audited. |
| `analyze_results.py`, `plots.py` | Descriptive summaries and per-run plots from audited CSVs. |
| `verify_invariants.py` | Manual invariant checks, also collected by pytest. |
| `phase1_smoke.py` | Tiny offline consensus run with persisted audit data; no downloads. |
| `tests/` | Regression and property tests. |
| `README.md` | Install, test, smoke and paper-run commands for a new user. |
| `requirements.txt` | The exact tested environment; not a resolved lock file. |
| `audit.md` | The Phase 1/2 audit. Historical record of findings. |
| `PHASE1.md` | Historical notes. Describes superseded behaviour; not current documentation. |

## Dataset conventions

Three datasets, all loaded through `load_dataset_records`:

- **HotpotQA** — `hotpot_qa`, config `distractor`, split `validation`. Context is
  parallel `title` / `sentences` lists; gold titles come from
  `supporting_facts.title`.
- **2WikiMultiHopQA** — `framolfese/2WikiMultihopQA`, same schema as HotpotQA.
- **MuSiQue** — `dgslibisey/MuSiQue`, file `musique_ans_v1.0_dev.jsonl`.
  Paragraphs carry `is_supporting`; gold titles are the titles of supporting
  paragraphs. The loader resolves that raw file with `hf_hub_download` at the
  pinned revision (default `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`;
  `local_files_only` when `HF_HUB_OFFLINE`/`HF_DATASETS_OFFLINE` is set) and
  reads the local path with `load_dataset("json", ...)`. It does not call
  `load_dataset("dgslibisey/MuSiQue", data_files=...)`: under datasets 5.0.1
  that call keys its processed cache on the resolved `hf://` URLs online and on
  the raw `data_files` mapping offline, so an offline load never finds the cache
  an online load built. Provenance records repository, revision, source filename
  and raw-file SHA-256 (`15fa6379…`, equal to the Hub LFS oid); the `datasets`
  fingerprint of the local json load is recorded separately and is not the
  source identity. The old repo-level fingerprint `9eaf7aa381348ab4` belongs to
  the previous loading path.

Support is read per paragraph where the source provides it and by title
otherwise. `paragraph_support` returns one flag per paragraph: MuSiQue's
`is_supporting` list directly, and for HotpotQA / 2Wiki the test
`title in gold_titles`, so under a title-only source every paragraph beneath a
supporting title counts as support. A title set cannot represent MuSiQue's
labels: roughly half of its dev examples repeat a title, and collapsing them
would both discard a supporting paragraph and mark a same-titled non-supporting
one as gold.

Candidate construction (`_assemble_candidates`), in order:

1. Raise if the support flags do not match the paragraph count.
2. Raise if any gold title is absent from the example's paragraphs.
3. Raise if no paragraph is support.
4. Identify a passage by `(title, text)`. Paragraphs sharing a title are
   distinct passages and both are kept. A byte-identical repeat collapses to one
   slot; a repeat whose support flag disagrees raises.
5. Raise if the number of gold passages exceeds the candidate count.
6. Take all gold, then same-query distractors in seeded shuffled order.
7. **Cross-query distractors** pad only when the query's own paragraphs are
   exhausted. That pool is built from non-support paragraphs of all scanned
   examples; titles already chosen (or gold for this query) are excluded, and
   each padded title supplies at most one passage.
8. Raise if the target candidate count still cannot be reached.
9. Shuffle the chosen slots with the seeded generator, then emit
   `"{title}: {text}"` strings and a `relevance` vector keyed by the chosen
   `(title, text)` pairs.

Gold-preservation rules: every referenced supporting passage reaches the pool, or
the example fails loudly. Nothing is dropped to make a pool fit. The RNG is not
consumed before a validation failure, so failures are deterministic.

The candidate slot index is the agent identity and is fixed across hops. This is
candidate-pool reranking, not corpus retrieval.

## Retriever conventions

All three are 384-dimensional, so `proj_dim = 384` and no random projection is
applied in the certificate path. Embeddings are L2-normalized after pooling, and
`max_seq_len` is 96.

| Checkpoint | Query format | Passage format | Pooling |
| --- | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | plain text | plain text | masked mean |
| `intfloat/e5-small-v2` | `query: {text}` | `passage: {text}` | masked mean |
| `BAAI/bge-small-en-v1.5` | `Represent this sentence for searching relevant passages: {text}` | plain text | CLS |

An unknown checkpoint silently falls back to identity formatting and mean
pooling. `adapter_metadata` only inspects an already-cached snapshot; it never
downloads, and it reports a metadata comparison, not encoder equivalence. Two of
the three checkpoints ship no Sentence-Transformers pooling config here, so it
reports `pooling_status: unvalidated` for them. That is a statement about missing
files. `tests/test_adapters.py` checks the adapters against the real encoders'
forward passes instead, and skips a checkpoint that is not cached rather than
downloading it.

At the preparation boundary a single embedding has shape `(proj_dim,)`.
`_build_agent_shards` validates the batch shapes `(n_queries, proj_dim)` and
`(n_queries * n_agents, proj_dim)` and raises on anything else. Do not add
`squeeze` calls to paper over a shape mismatch; fix it at this boundary.

## Tests

    .venv/Scripts/python.exe -B -m pytest -q --hypothesis-show-statistics
    .venv/Scripts/python.exe -B verify_invariants.py
    .venv/Scripts/python.exe -m ruff check .

`pytest` collects `tests/` and `verify_invariants.py`. `tests/conftest.py` forces
offline HuggingFace mode and a derandomized Hypothesis profile at 100 examples.
Ruff is configured in `pyproject.toml` with `E`, `F` and `W`, ignoring `E501`,
`E741` and `E731`; it is a check, not a formatter. Do not reformat code to satisfy
it.

What the suites protect:

- `tests/test_certificate_properties.py` — randomized soundness: the
  per-candidate bounds really contain the score changes; `global => pairwise`; a
  pairwise certificate implies the top-k set is unchanged; an ordered certificate
  implies the ranking is unchanged.
- `tests/test_numerical_bounds.py` — float64 envelopes against exact rational
  arithmetic; the operator-norm enclosure bounds every Rayleigh quotient;
  zero/tied margins are rejected; non-finite input fails closed; a tiny positive
  slack survives reporting.
- `tests/test_bound_diagnostics.py` — a large bound on one candidate cannot mask
  a per-candidate bound failure on another.
- `tests/test_datasets.py` — gold-preservation failures for all three source
  schemas, and determinism of candidate assembly.
- `tests/test_musique_loader.py` — MuSiQue is read from the pinned raw file in
  both modes, with identical rows, `is_supporting` labels and file hash; an
  uncached file fails clearly offline. The real-file checks skip when the file is
  not cached.
- `tests/test_retrieval.py` — the real preparation path, all 3 datasets x 3
  retrievers, through to a certificate; and rejection of ambiguous embedding
  shapes.
- `tests/test_phase1.py` — decision immutability, decision-before-mutation
  ordering, label isolation, noise rejection, CLI separation of graph neighbours
  and retrieval depth, CSV provenance columns.
- `tests/test_results.py` — altered CSVs, altered or missing run artifacts, and
  legacy CSVs are rejected or explicitly marked unaudited.
- `tests/test_packaging.py` — every module reachable from an entry point is
  declared in `pyproject.toml`, and caches/results/runs are not packaged.
- `tests/test_adapters.py` — each retriever's formatting, pooling, normalization
  and 384-dimensional output checked against the real cached encoder; skipped for
  an uncached checkpoint.
- `tests/test_legacy_outputs.py` — the historical Fisher/SIR writers refuse to
  replace an existing CSV or figure, and default to a fresh directory.
- `tests/test_artifact_index.py` — a verified run with no loaded encoder is
  labelled `development_fixture`, a sweep parent is a command record, and
  retained results stay `legacy_unverified`.

`tests/conftest.py` also fails any test that leaves a file behind in the
repository's `runs/`, `results/` or `figures/`: a run directory is provenance, and
a test fixture there is indistinguishable from a real certificate run in the
inventory. Tests that call `run_topk_stability_experiment` or a legacy writer must
point their output at a `tmp_path`.

## Working rules

- Inspect `git diff` and `git status` before modifying code. Uncommitted work may
  be a previous session's correct in-progress fix.
- Make minimal changes. Do not mass-revert or mass-reformat.
- Add the smallest regression test that would have caught each bug fixed.
- Never weaken a test, tolerance, or trial count to make code pass.
- Never silently change a mathematical assumption. If a bound, comparison, or
  inequality changes, say so explicitly.
- Never modify historical result artifacts (`results/`, `figures/`,
  `protocol_corrected_run.log`, existing `runs/`).
- Never run expensive paper experiments unless explicitly asked. Small smoke runs
  only.
- Preserve provenance. New run artifacts go in a fresh run directory with a
  manifest; existing manifests and hashes are not rewritten.
- Report uncertainty rather than guessing. If something cannot be verified from
  code or tests, say it is unverified.

## Code style

This is a small academic Python project and should read like one.

Use ordinary human naming and straightforward functions with domain-specific
names.

Avoid:

- decorative comments, banner comments, emojis
- generated-sounding prose
- comments that merely restate the code
- unnecessary abstractions, managers, controllers, engines
- filenames such as `enhanced_*`, `advanced_*`, `smart_*`, `ultimate_*`,
  `final_*`, `*_v2`
- phase-number filenames, unless the file is genuinely a historical document
  (`PHASE1.md` is one)
