from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dataset_loaders
import evaluation_harness as harness
from model_adapters import format_passage, format_query, known_retrievers


@pytest.mark.parametrize("dataset", dataset_loaders.known_datasets())
@pytest.mark.parametrize("retriever", known_retrievers())
def test_loader_to_certificate(dataset, retriever, tmp_path, monkeypatch):
    paragraphs = [("A", "support a"), ("B", "support b"), ("C", "negative")]
    if dataset == "musique":
        source = {"question": "question", "paragraphs": [
            {"title": t, "paragraph_text": text, "is_supporting": t != "C"}
            for t, text in paragraphs]}
    else:
        source = {"question": "question", "context": {
            "title": [t for t, _ in paragraphs], "sentences": [[text] for _, text in paragraphs]},
            "supporting_facts": {"title": ["A", "B"]}}
    raw_file = tmp_path / "dev.jsonl"
    raw_file.write_text("")
    monkeypatch.setattr(dataset_loaders, "_resolve_musique_file", lambda revision: str(raw_file))
    monkeypatch.setattr(dataset_loaders, "load_dataset", lambda *a, **kw: [source])
    texts_seen = []

    class Tokenizer:
        pad_token = "pad"

        def __call__(self, texts, **kwargs):
            assert kwargs["max_length"] == 96
            texts_seen.extend(texts)
            return {"input_ids": torch.tensor([[len(t) % 7 + 1, 2, 0] for t in texts]),
                    "attention_mask": torch.tensor([[1, 1, 0]] * len(texts))}

    class Encoder(torch.nn.Module):
        config = SimpleNamespace(_commit_hash=None)

        def forward(self, input_ids, attention_mask):
            x = input_ids.float()
            return SimpleNamespace(last_hidden_state=torch.stack((x, x * .3 + 1), dim=-1))

    monkeypatch.setattr(harness.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(harness.AutoModel, "from_pretrained", lambda *a, **kw: Encoder())
    cfg = harness.TopKStabilityConfig(dataset=dataset, lm_checkpoint=retriever,
        n_agents=3, proj_dim=2, k=2, n_hops=1, n_eval_queries=1, run_dir=str(tmp_path))
    records, shards, states = harness._prepare_hotpotqa_run(
        cfg, bias=False, identity_init=True, dataset=dataset)
    assert texts_seen[0] == format_query(retriever, "question")
    assert set(texts_seen[1:]) == {format_passage(retriever, f"{t}: {x}") for t, x in paragraphs}
    assert records[0]["relevance"].sum() == 2
    for shard in shards:
        assert shard[0]["e_q"].shape == shard[0]["e_doc"].shape == (2,)
        assert torch.allclose(shard[0]["e_q"].norm(), torch.tensor(1.))
    rows = []
    history = harness.run_topk_stability_experiment(cfg, records, shards, states, rows)
    assert len(history) == len(rows) == 1
    assert rows[0]["certificate_kind"] == "float64_guarded"
    assert rows[0]["per_document_bound_holds"]
    assert history[0].global_certificate_violations == history[0].pairwise_certificate_violations == 0
    assert history[0].ordered_certificate_violations == 0
    assert next(tmp_path.glob("*/completed.json")).is_file()


@pytest.mark.parametrize("shape", [(2,), (1, 1, 2), (1, 3)])
def test_preparation_rejects_ambiguous_embedding_shapes(shape):
    cfg = harness.TopKStabilityConfig(n_agents=3, proj_dim=2, k=2)
    with pytest.raises(ValueError, match="embedding.*shape"):
        harness._build_agent_shards(cfg, [{"relevance": np.ones(3)}],
                                    torch.ones(shape), torch.ones(3, 2))
