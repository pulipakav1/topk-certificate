"""What each retriever actually does, checked against its own forward pass.

`adapter_metadata` only compares cached Sentence-Transformers metadata files, and
two of the three checkpoints do not ship them here, so it reports `unvalidated`.
That is a statement about missing files, not about the adapter. These checks read
the behaviour off the implementation and the real encoder instead, and skip when
a checkpoint is not in the local cache rather than reaching the network.
"""

import os
from pathlib import Path

import pytest
import torch

import evaluation_harness as harness
from model_adapters import (
    format_passage,
    format_query,
    known_retrievers,
    pool_hidden_states,
    pooling_strategy,
)

EXPECTED = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "query": "what is a spectral norm?",
        "passage": "Norms: the operator 2-norm is the largest singular value.",
        "pooling": "mean",
    },
    "intfloat/e5-small-v2": {
        "query": "query: what is a spectral norm?",
        "passage": "passage: Norms: the operator 2-norm is the largest singular value.",
        "pooling": "mean",
    },
    "BAAI/bge-small-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: "
                 "what is a spectral norm?",
        "passage": "Norms: the operator 2-norm is the largest singular value.",
        "pooling": "cls",
    },
}
QUERY = "what is a spectral norm?"
PASSAGE = "Norms: the operator 2-norm is the largest singular value."
PROJ_DIM = 384


def snapshot(checkpoint):
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    repo = hub / ("models--" + checkpoint.replace("/", "--"))
    reference = repo / "refs" / "main"
    if not reference.is_file():
        return None
    directory = repo / "snapshots" / reference.read_text().strip()
    return directory if directory.is_dir() else None


@pytest.mark.parametrize("checkpoint", known_retrievers())
def test_formatting_and_pooling_choice(checkpoint):
    expected = EXPECTED[checkpoint]
    assert format_query(checkpoint, QUERY) == expected["query"]
    assert format_passage(checkpoint, PASSAGE) == expected["passage"]
    assert pooling_strategy(checkpoint) == expected["pooling"]


@pytest.mark.parametrize("checkpoint", known_retrievers())
def test_pooling_operation_matches_its_definition(checkpoint):
    """CLS takes token 0; mean averages only unmasked tokens."""
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_hidden_states(hidden, mask, checkpoint)
    if pooling_strategy(checkpoint) == "cls":
        expected = torch.tensor([[1., 2.]])
    else:
        expected = torch.tensor([[2., 3.]])
    assert torch.allclose(pooled, expected)


@pytest.mark.parametrize("checkpoint", known_retrievers())
def test_real_encoder_output_matches_the_adapter(checkpoint):
    """Load the cached encoder and recompute pooling and normalization by hand."""
    if snapshot(checkpoint) is None:
        pytest.skip(f"{checkpoint} is not in the local Hugging Face cache")
    tokenizer = harness.AutoTokenizer.from_pretrained(checkpoint)
    encoder = harness.AutoModel.from_pretrained(checkpoint)
    encoder.eval()
    cfg = harness.TopKStabilityConfig(lm_checkpoint=checkpoint, proj_dim=PROJ_DIM)

    texts = [format_query(checkpoint, QUERY), format_passage(checkpoint, PASSAGE)]
    embeddings = harness._embed_texts_retrieval(encoder, tokenizer, texts, cfg)

    assert embeddings.shape == (2, PROJ_DIM)
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2), atol=1e-5)

    with torch.no_grad():
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=cfg.max_seq_len)
        hidden = encoder(**encoded).last_hidden_state
    assert encoded["input_ids"].shape[1] <= cfg.max_seq_len
    if pooling_strategy(checkpoint) == "cls":
        manual = hidden[:, 0]
    else:
        weights = encoded["attention_mask"].unsqueeze(-1).float()
        manual = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.)
    manual = manual / manual.norm(dim=-1, keepdim=True)
    assert torch.allclose(embeddings, manual, atol=1e-6)


@pytest.mark.parametrize("checkpoint", known_retrievers())
def test_real_encoder_hidden_size_is_the_certificate_dimension(checkpoint):
    if snapshot(checkpoint) is None:
        pytest.skip(f"{checkpoint} is not in the local Hugging Face cache")
    encoder = harness.AutoModel.from_pretrained(checkpoint)
    assert encoder.config.hidden_size == PROJ_DIM


def test_query_and_passage_formatting_differ_only_where_intended():
    """A prefix is added, and the text it wraps is never altered."""
    for checkpoint in known_retrievers():
        for formatted in (format_query(checkpoint, QUERY), format_passage(checkpoint, PASSAGE)):
            assert formatted.endswith(QUERY) or formatted.endswith(PASSAGE)
