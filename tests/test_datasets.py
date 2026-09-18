import numpy as np
import pytest

import dataset_loaders as loaders
from dataset_loaders import RawExample, _assemble_candidates


@pytest.mark.parametrize("paragraphs,gold,supporting,message", [
    ([("A", "a"), ("C", "c")], {"A", "B"}, None, "missing.*B"),
    ([("A", "a"), ("B", "b"), ("C", "c")], {"A", "B", "C"}, None, "3 gold passages"),
    ([("A", "a"), ("C", "c")], set(), None, "no supporting passages"),
    # The identical passage cannot be both support and not support.
    ([("A", "same"), ("A", "same")], {"A"}, [True, False], "identical passage 'A'"),
    ([("A", "a")], {"A"}, [True, False], "1 paragraphs"),
])
def test_gold_preservation_fails_before_sampling(paragraphs, gold, supporting, message):
    rng = np.random.default_rng(42)
    state = rng.bit_generator.state
    example = RawExample("question", gold, paragraphs, supporting=supporting)
    with pytest.raises(ValueError, match=message):
        _assemble_candidates(example, 2, [], rng)
    assert rng.bit_generator.state == state


def test_paragraphs_sharing_a_title_are_distinct_passages():
    """MuSiQue repeats titles; a title set would drop or mislabel the repeat."""
    example = RawExample("q", {"A"}, [("A", "support one"), ("A", "support two")],
                         supporting=[True, True])
    docs, relevance = _assemble_candidates(example, 2, [], np.random.default_rng(0))
    assert set(docs) == {"A: support one", "A: support two"}
    assert relevance.sum() == 2

    # The non-supporting paragraph must not inherit gold from its title.
    example = RawExample("q", {"A"}, [("A", "support"), ("A", "not support")],
                         supporting=[True, False])
    docs, relevance = _assemble_candidates(example, 2, [], np.random.default_rng(0))
    assert dict(zip(docs, relevance)) == {"A: support": 1., "A: not support": 0.}


def test_identical_repeated_passage_collapses_to_one_slot():
    example = RawExample("q", {"A"}, [("A", "same"), ("A", "same"), ("B", "b")],
                         supporting=[True, True, False])
    docs, relevance = _assemble_candidates(example, 2, [], np.random.default_rng(0))
    assert sorted(docs) == ["A: same", "B: b"]
    assert relevance.sum() == 1


def test_title_only_sources_treat_every_paragraph_under_a_gold_title_as_support():
    """HotpotQA/2Wiki give supporting titles only, so no repeat can be dropped."""
    example = RawExample("q", {"A"}, [("A", "first"), ("A", "second")])
    docs, relevance = _assemble_candidates(example, 2, [], np.random.default_rng(0))
    assert set(docs) == {"A: first", "A: second"}
    assert relevance.tolist() == [1., 1.]


def test_multiple_gold_padding_and_exact_count_are_deterministic():
    example = RawExample("q", {"A", "B"}, [("A", "a"), ("B", "b"), ("C", "c")])
    pool = [("D", "d"), ("E", "e"), ("F", "f")]
    first = _assemble_candidates(example, 5, pool, np.random.default_rng(42))
    second = _assemble_candidates(example, 5, pool, np.random.default_rng(42))
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == 5 and first[1].sum() == 2
    assert {p for p, label in zip(*first) if label} == {"A: a", "B: b"}


def test_insufficient_pool_fails_clearly():
    with pytest.raises(ValueError, match="candidate count"):
        _assemble_candidates(RawExample("q", {"A"}, [("A", "a")]),
                             3, [], np.random.default_rng(0))


def test_a_missing_processed_cache_says_to_load_once_online(monkeypatch):
    def missing_cache(*args, **kwargs):
        raise ValueError("Couldn't find cache for hotpot_qa for config 'distractor'")

    monkeypatch.setattr(loaders, "load_dataset", missing_cache)
    with pytest.raises(ValueError, match="Load it once with network access") as error:
        loaders.load_dataset_records("hotpotqa", 1, 2)
    assert "version" not in str(error.value).split("Original error")[0]


@pytest.mark.parametrize("dataset", loaders.known_datasets())
def test_other_loader_errors_are_not_rewritten(dataset, monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("some other dataset problem")

    source = tmp_path / "dev.jsonl"
    source.write_text("")
    monkeypatch.setattr(loaders, "_resolve_musique_file", lambda revision: str(source))
    monkeypatch.setattr(loaders, "load_dataset", broken)
    with pytest.raises(ValueError, match="^some other dataset problem$"):
        loaders.load_dataset_records(dataset, 1, 2)


@pytest.mark.parametrize("dataset", loaders.known_datasets())
def test_dataset_adapter_keeps_both_paragraphs_of_a_repeated_title(dataset, monkeypatch, tmp_path):
    """A repeated title reaches the pool intact, labelled as the source labels it."""
    if dataset == "musique":
        source = {"question": "q", "paragraphs": [
            {"title": "A", "paragraph_text": "negative", "is_supporting": False},
            {"title": "A", "paragraph_text": "support", "is_supporting": True}]}
        expected = {"A: negative": 0., "A: support": 1.}
    else:
        source = {"question": "q", "context": {"title": ["A", "A"],
                  "sentences": [["negative"], ["support"]]}, "supporting_facts": {"title": ["A"]}}
        # Only a supporting title is given, so both paragraphs count as support.
        expected = {"A: negative": 1., "A: support": 1.}
    raw_file = tmp_path / "dev.jsonl"
    raw_file.write_text("")
    monkeypatch.setattr(loaders, "_resolve_musique_file", lambda revision: str(raw_file))
    monkeypatch.setattr(loaders, "load_dataset", lambda *a, **kw: [source])
    records = loaders.load_dataset_records(dataset, 1, 2)
    assert len(records) == 1
    assert dict(zip(records[0]["paragraphs"], records[0]["relevance"])) == expected
