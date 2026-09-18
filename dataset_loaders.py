"""dataset loaders"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
except ImportError:
    hf_hub_download = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# common shape
# --------------------------------------------------------------------------
#
# question text
# gold titles
# all paragraphs

@dataclass
class RawExample:
    question: str
    gold_titles: set
    paragraphs: List[Tuple[str, str]]  # (title, text)
    source_metadata: dict = field(default_factory=dict)
    supporting: Optional[List[bool]] = None  # per paragraph, when the source says


def paragraph_support(ex: RawExample) -> List[bool]:
    """Which paragraphs are support, one flag per paragraph.

    MuSiQue marks support per paragraph and repeats titles across different
    paragraphs, so a title set cannot represent its labels: it would both lose a
    supporting paragraph and mark a same-titled non-supporting one as gold.
    HotpotQA and 2Wiki only give supporting titles, so there the flags are derived
    by title and every paragraph under a supporting title counts as support.
    """
    if ex.supporting is None:
        return [title in ex.gold_titles for title, _ in ex.paragraphs]
    if len(ex.supporting) != len(ex.paragraphs):
        raise ValueError(f"Example {ex.question!r} has {len(ex.supporting)} support flags "
                         f"for {len(ex.paragraphs)} paragraphs")
    return [bool(flag) for flag in ex.supporting]


# --------------------------------------------------------------------------
# hotpotqa style
# --------------------------------------------------------------------------
#
# same schema
# reused for 2wiki

def _iter_hotpotqa_style_raw(
    repo_id: str, slice_str: str, config: Optional[str] = None, revision=None
) -> Iterable[RawExample]:
    """hotpotqa style"""
    kwargs = {"split": slice_str, "revision": revision}
    ds = load_dataset(repo_id, config, **kwargs) if config else load_dataset(repo_id, **kwargs)
    metadata = {"repository": repo_id, "split": slice_str, "revision": revision,
                "fingerprint": getattr(ds, "_fingerprint", None), "configuration": config}
    for ex in ds:
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        if len(titles) != len(sentences):
            raise ValueError(f"Example {ex['question']!r} has mismatched context titles/sentences")
        paragraphs = [(titles[j], " ".join(sentences[j])) for j in range(len(titles))]
        gold_titles = set(ex["supporting_facts"]["title"])
        yield RawExample(ex["question"], gold_titles, paragraphs, metadata)


def _iter_hotpotqa_raw(slice_str: str, revision=None) -> Iterable[RawExample]:
    """hotpotqa raw"""
    yield from _iter_hotpotqa_style_raw("hotpot_qa", slice_str, config="distractor", revision=revision)


def _iter_2wiki_raw(slice_str: str, revision=None) -> Iterable[RawExample]:
    """2wiki raw"""
    yield from _iter_hotpotqa_style_raw("framolfese/2WikiMultihopQA", slice_str, revision=revision)


# --------------------------------------------------------------------------
# musique
# --------------------------------------------------------------------------
#
# own schema
# is_supporting flag

MUSIQUE_REPOSITORY = "dgslibisey/MuSiQue"
MUSIQUE_DEV_FILE = "musique_ans_v1.0_dev.jsonl"
MUSIQUE_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"


def _offline_mode() -> bool:
    """True when either HuggingFace offline switch is set."""
    return any(os.environ.get(name, "").strip().upper() in {"1", "ON", "YES", "TRUE"}
               for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_musique_file(revision: str) -> str:
    """Local path of the MuSiQue dev JSONL at `revision`, from the Hub cache.

    Loading the repository with `load_dataset(repo, data_files=...)` is not used:
    datasets 5.0.1 keys the processed cache on the resolved hf:// URLs online but
    on the raw data_files mapping offline, so an offline load never finds what an
    online load built. The raw snapshot file has the same identity in both modes.
    """
    if hf_hub_download is None:
        raise ImportError("The `huggingface_hub` package is required to load MuSiQue.")
    offline = _offline_mode()
    try:
        return hf_hub_download(MUSIQUE_REPOSITORY, MUSIQUE_DEV_FILE, repo_type="dataset",
                               revision=revision, local_files_only=offline)
    except LocalEntryNotFoundError as exc:
        if not offline:
            raise
        raise ValueError(
            f"{MUSIQUE_REPOSITORY}/{MUSIQUE_DEV_FILE} at revision {revision} is not in the "
            f"HuggingFace Hub cache and offline mode is set. Download it once with network "
            f"access (unset HF_HUB_OFFLINE/HF_DATASETS_OFFLINE), then rerun offline."
        ) from exc


def _iter_musique_raw(slice_str: str, revision=None) -> Iterable[RawExample]:
    """musique raw"""
    revision = revision or MUSIQUE_REVISION
    path = _resolve_musique_file(revision)
    ds = load_dataset("json", data_files={"validation": path}, split=slice_str)
    # The fingerprint belongs to this local json load; the source identity is the
    # repository, revision, filename and raw file hash.
    metadata = {"repository": MUSIQUE_REPOSITORY, "split": slice_str, "revision": revision,
                "source_file": MUSIQUE_DEV_FILE, "source_sha256": _file_sha256(path),
                "loader": "hf_hub_download+json",
                "fingerprint": getattr(ds, "_fingerprint", None)}
    for ex in ds:
        paragraphs = [(p["title"], p["paragraph_text"]) for p in ex["paragraphs"]]
        supporting = [bool(p["is_supporting"]) for p in ex["paragraphs"]]
        gold_titles = {title for (title, _), sup in zip(paragraphs, supporting) if sup}
        yield RawExample(ex["question"], gold_titles, paragraphs, metadata,
                         supporting=supporting)


_DATASET_LOADERS = {
    "hotpotqa": (_iter_hotpotqa_raw, "validation"),
    "musique": (_iter_musique_raw, "validation"),
    "2wikimultihopqa": (_iter_2wiki_raw, "validation"),
}


def known_datasets() -> Tuple[str, ...]:
    """dataset list"""
    return tuple(_DATASET_LOADERS.keys())


# --------------------------------------------------------------------------
# candidate assembly
# --------------------------------------------------------------------------
#
# ambiguous support is rejected
# same-query fill
# cross-query pad

def _shuffled(items: list, rng: np.random.Generator) -> list:
    """order shuffle"""
    idx = rng.permutation(len(items))
    return [items[i] for i in idx]


def _assemble_candidates(
    ex: RawExample, n_candidates: int, pool: List[Tuple[str, str]], rng: np.random.Generator
) -> Tuple[List[str], np.ndarray]:
    """assemble docs"""
    if not isinstance(n_candidates, (int, np.integer)) or n_candidates <= 0:
        raise ValueError("candidate count must be a positive integer")
    support = paragraph_support(ex)
    missing = ex.gold_titles - {title for title, _ in ex.paragraphs}
    if missing:
        raise ValueError(f"Example {ex.question!r} is missing supporting titles: {sorted(missing)}")
    if not any(support):
        raise ValueError(f"Example {ex.question!r} has no supporting passages")
    # A passage is (title, text). Two paragraphs sharing a title are different
    # passages and both are kept; only a byte-identical repeat is a duplicate, and
    # only a repeat whose support flag disagrees is genuinely ambiguous.
    seen, gold, distractors = {}, [], []
    for (title, text), is_gold in zip(ex.paragraphs, support, strict=True):
        key = (title, text)
        if key in seen:
            if seen[key] != is_gold:
                raise ValueError(f"Example {ex.question!r} labels the identical passage {title!r} "
                                 "both supporting and non-supporting; support preservation "
                                 "cannot be guaranteed")
            continue
        seen[key] = is_gold
        (gold if is_gold else distractors).append((title, text))

    if len(gold) > n_candidates:
        raise ValueError(
            f"Example {ex.question!r} has {len(gold)} gold passages but "
            f"candidate_slots={n_candidates}; increase the candidate count."
        )

    distractors = _shuffled(distractors, rng)
    n_needed = max(n_candidates - len(gold), 0)
    chosen = gold + distractors[:n_needed]

    if len(chosen) < n_candidates:
        # same-query pool exhausted -- pad with distractors from other queries,
        # one passage per title so a padded title never shadows another passage
        used = {t for t, _ in chosen} | ex.gold_titles
        for title, text in _shuffled([p for p in pool if p[0] not in used], rng):
            if len(chosen) >= n_candidates:
                break
            if title in used:
                continue
            used.add(title)
            chosen.append((title, text))

    if len(chosen) != n_candidates:
        raise ValueError(f"Example {ex.question!r} cannot reach candidate count {n_candidates}; "
                         f"only {len(chosen)} passages are available")
    gold_passages = {(t, x) for t, x in gold}
    chosen = _shuffled(chosen, rng)
    relevance = np.array([1.0 if (t, x) in gold_passages else 0.0 for t, x in chosen],
                         dtype=np.float64)
    paragraphs = [f"{t}: {x}".strip() for t, x in chosen]
    return paragraphs, relevance


# --------------------------------------------------------------------------
# common loader
# --------------------------------------------------------------------------

def load_dataset_records(
    dataset: str,
    n_eval_queries: int,
    n_candidates: int,
    seed: int = 42,
    dataset_slice: Optional[str] = None,
    revision: Optional[str] = None,
) -> List[dict]:
    """common loader"""
    if not _DATASETS_AVAILABLE:
        raise ImportError("The `datasets` package is required for this pipeline (pip install datasets).")
    if dataset not in _DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(_DATASET_LOADERS)}")
    if not isinstance(n_eval_queries, (int, np.integer)) or n_eval_queries <= 0:
        raise ValueError("n_eval_queries must be a positive integer")
    raw_iter_fn, default_split = _DATASET_LOADERS[dataset]

    scan_n = max(int(n_eval_queries * 1.2) + 50, n_eval_queries + 50)
    slice_str = dataset_slice or f"{default_split}[:{scan_n}]"

    logger.info("Loading %s (%s) ...", dataset, slice_str)
    try:
        raw_examples = list(raw_iter_fn(slice_str, revision=revision))
    except ValueError as exc:
        if "Couldn't find cache" not in str(exc):
            raise
        # Repository loads (HotpotQA, 2Wiki) are served offline only from a processed
        # cache the offline lookup can find; one online load at the same revision
        # builds it. MuSiQue reads its raw snapshot file and does not come here.
        raise ValueError(
            f"{dataset} has no processed `datasets` cache that offline mode can find for "
            f"this configuration and revision. Load it once with network access (unset "
            f"HF_DATASETS_OFFLINE/HF_HUB_OFFLINE), then rerun offline. Original error: {exc}"
        ) from exc

    # cross-query distractor pool, used only to pad a query short of n_candidates
    pool: List[Tuple[str, str]] = []
    for ex in raw_examples:
        for (title, text), is_gold in zip(ex.paragraphs, paragraph_support(ex), strict=True):
            if not is_gold:
                pool.append((title, text))

    rng = np.random.default_rng(seed)
    records: List[dict] = []
    for ex in raw_examples:
        if len(records) >= n_eval_queries:
            break
        paragraphs, relevance = _assemble_candidates(ex, n_candidates, pool, rng)
        records.append({"question": ex.question, "paragraphs": paragraphs, "relevance": relevance,
                        "source_metadata": ex.source_metadata})

    if len(records) < n_eval_queries:
        logger.warning(
            "Only found %d usable %s examples (wanted %d) within %s.",
            len(records), dataset, n_eval_queries, slice_str,
        )
    logger.info("Selected %d %s queries, %d candidate paragraphs each.", len(records), dataset, n_candidates)
    return records
