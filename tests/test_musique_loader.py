"""MuSiQue is read from the pinned raw Hub file, the same way online and offline."""

import json

import pytest

import dataset_loaders as loaders

DEV_ROWS = 2417
DEV_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"


def set_offline(monkeypatch, offline):
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if offline:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("offline", [False, True])
def test_file_resolution_follows_offline_mode(offline, monkeypatch):
    calls = []
    monkeypatch.setattr(loaders, "hf_hub_download",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or "local.jsonl")
    set_offline(monkeypatch, offline)
    assert loaders._resolve_musique_file(loaders.MUSIQUE_REVISION) == "local.jsonl"
    assert calls == [(("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl"),
                      {"repo_type": "dataset", "revision": "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
                       "local_files_only": offline})]


def test_musique_loads_the_local_file_not_the_repository(monkeypatch, tmp_path):
    raw_file = tmp_path / "dev.jsonl"
    raw_file.write_bytes(b"raw")
    revisions, loads = [], []
    monkeypatch.setattr(loaders, "_resolve_musique_file",
                        lambda revision: revisions.append(revision) or str(raw_file))
    monkeypatch.setattr(loaders, "load_dataset", lambda *a, **kw: loads.append((a, kw)) or [])
    assert list(loaders._iter_musique_raw("validation[:3]")) == []
    assert revisions == [loaders.MUSIQUE_REVISION]
    assert loads == [(("json",), {"data_files": {"validation": str(raw_file)},
                                  "split": "validation[:3]"})]


def test_uncached_pinned_file_fails_clearly_offline(monkeypatch):
    set_offline(monkeypatch, True)
    with pytest.raises(ValueError, match="is not in the HuggingFace Hub cache and offline mode"):
        loaders.load_dataset_records("musique", 1, 2, revision="0" * 40)


@pytest.fixture
def cached_dev_file(monkeypatch):
    set_offline(monkeypatch, True)
    try:
        return loaders._resolve_musique_file(loaders.MUSIQUE_REVISION)
    except ValueError:
        pytest.skip("pinned MuSiQue dev file is not cached")


def rows(examples):
    return [(ex.question, ex.paragraphs, ex.supporting, ex.gold_titles) for ex in examples]


def test_online_and_offline_paths_read_the_same_pinned_file(cached_dev_file, monkeypatch):
    # With the offline switches unset the loader asks the Hub, not the cache only.
    # conftest keeps huggingface_hub itself offline, so this resolves the cached
    # snapshot through the online call without touching the network.
    offline = list(loaders._iter_musique_raw("validation"))
    set_offline(monkeypatch, False)
    online_path = loaders._resolve_musique_file(loaders.MUSIQUE_REVISION)
    online = list(loaders._iter_musique_raw("validation"))

    assert online_path == cached_dev_file
    assert len(offline) == len(online) == DEV_ROWS
    assert rows(online) == rows(offline)
    for examples in (online, offline):
        source = examples[0].source_metadata
        assert source["repository"] == "dgslibisey/MuSiQue"
        assert source["revision"] == loaders.MUSIQUE_REVISION
        assert source["source_file"] == "musique_ans_v1.0_dev.jsonl"
        assert source["source_sha256"] == DEV_SHA256
        assert source["loader"] == "hf_hub_download+json"
        assert source["fingerprint"] != "9eaf7aa381348ab4"


def test_support_labels_are_the_raw_is_supporting_flags(cached_dev_file):
    with open(cached_dev_file, encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    loaded = list(loaders._iter_musique_raw("validation"))
    assert len(raw) == len(loaded) == DEV_ROWS
    for source, example in zip(raw, loaded):
        assert example.question == source["question"]
        assert example.paragraphs == [(p["title"], p["paragraph_text"]) for p in source["paragraphs"]]
        assert example.supporting == [bool(p["is_supporting"]) for p in source["paragraphs"]]
        assert loaders.paragraph_support(example) == example.supporting
