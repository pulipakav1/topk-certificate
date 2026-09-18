"""model adapters"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import hashlib
import json
import os
from pathlib import Path

Formatter = Callable[[str], str]


def _identity(text: str) -> str:
    """no prefix"""
    return text


def _e5_query(text: str) -> str:
    """e5 query"""
    return f"query: {text}"


def _e5_passage(text: str) -> str:
    """e5 passage"""
    return f"passage: {text}"


def _bge_query(text: str) -> str:
    """bge query"""
    return f"Represent this sentence for searching relevant passages: {text}"


# --------------------------------------------------------------------------
# formatter registry
# --------------------------------------------------------------------------
#
# minilm: plain text
# e5: query/passage prefix
# bge: query-only prefix

_FORMATTERS: Dict[str, Tuple[Formatter, Formatter]] = {
    "sentence-transformers/all-MiniLM-L6-v2": (_identity, _identity),
    "intfloat/e5-small-v2": (_e5_query, _e5_passage),
    "BAAI/bge-small-en-v1.5": (_bge_query, _identity),
}


def format_query(checkpoint: str, text: str) -> str:
    """query text"""
    query_fn, _ = _FORMATTERS.get(checkpoint, (_identity, _identity))
    return query_fn(text)


def format_passage(checkpoint: str, text: str) -> str:
    """passage text"""
    _, passage_fn = _FORMATTERS.get(checkpoint, (_identity, _identity))
    return passage_fn(text)


def known_retrievers() -> Tuple[str, ...]:
    """model list"""
    return tuple(_FORMATTERS.keys())


# --------------------------------------------------------------------------
# pooling registry
# --------------------------------------------------------------------------
#
# minilm: mean pooling (sentence-transformers config)
# e5: mean pooling (attention-mask aware)
# bge: cls pooling

_POOLING: Dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "mean",
    "intfloat/e5-small-v2": "mean",
    "BAAI/bge-small-en-v1.5": "cls",
}


def pooling_strategy(checkpoint: str) -> str:
    """pool method"""
    return _POOLING.get(checkpoint, "mean")


def pool_hidden_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, checkpoint: str
) -> torch.Tensor:
    """Map (batch, tokens, dimension) states to (batch, dimension) embeddings."""
    if (hidden_states.ndim != 3 or attention_mask.ndim != 2 or
            hidden_states.shape[:2] != attention_mask.shape or hidden_states.shape[1] == 0):
        raise ValueError("Hidden states and attention mask must have matching batch/token dimensions")
    if not torch.isfinite(hidden_states).all() or not torch.isfinite(attention_mask).all():
        raise ValueError("Hidden states and attention mask must be finite")
    if pooling_strategy(checkpoint) == "cls":
        return hidden_states[:, 0]
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def adapter_metadata(checkpoint, *, max_seq_len=96, revision=None, cache_root=None):
    """Inspect cached official metadata only. Never calls a download API."""
    hub = Path(cache_root or os.environ.get("HF_HUB_CACHE") or
               (Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    repo = hub / ("models--" + checkpoint.replace("/", "--"))
    ref = repo / "refs" / (revision or "main")
    resolved = ref.read_text().strip() if ref.is_file() else revision
    snapshot = repo / "snapshots" / (resolved or "unavailable")
    evidence = {}
    metadata = {}
    for name in ("modules.json", "1_Pooling/config.json", "sentence_bert_config.json",
                 "config_sentence_transformers.json", "README.md"):
        path = snapshot / name
        if path.is_file():
            raw = path.read_bytes()
            evidence[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            if name.endswith(".json"):
                metadata[name] = json.loads(raw)
    pooling = metadata.get("1_Pooling/config.json", {})
    expected_key = "pooling_mode_mean_tokens" if pooling_strategy(checkpoint) == "mean" else "pooling_mode_cls_token"
    active = [key for key, value in pooling.items() if key.startswith("pooling_mode_") and value is True]
    pool_valid = active == [expected_key]
    normalization_valid = any(m.get("type", "").endswith(".Normalize") for m in metadata.get("modules.json", []))
    official_cap = metadata.get("sentence_bert_config.json", {}).get("max_seq_length")
    custom = official_cap is not None and official_cap != max_seq_len
    return {
        "checkpoint": checkpoint, "revision": resolved if snapshot.is_dir() else None,
        "status": "intentionally_custom" if custom else "unvalidated",
        "pooling": pooling_strategy(checkpoint), "pooling_status": "validated" if pool_valid else "unvalidated",
        "normalization": "L2 after pooling", "normalization_status": "validated" if normalization_valid else "unvalidated",
        "query_prefix": format_query(checkpoint, ""), "passage_prefix": format_passage(checkpoint, ""),
        "formatting_status": "unvalidated", "max_seq_len": max_seq_len,
        "official_max_seq_len": official_cap, "evidence": evidence,
        "scope": "metadata comparison only; no encoder equivalence or checkpoint quality claim",
    }
