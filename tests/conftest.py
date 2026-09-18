"""Offline deterministic CI profile; preserve all property trial counts."""

import os
from pathlib import Path

import pytest
from hypothesis import settings

# Set before any test module imports datasets/transformers, so no test can reach
# the network. conftest is imported first, so top-level imports here are safe.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

settings.register_profile("ci", derandomize=True, database=None, deadline=None, max_examples=100)
settings.load_profile("ci")

REPOSITORY = Path(__file__).resolve().parent.parent
OUTPUT_ROOTS = ("runs", "results", "figures")


@pytest.fixture(autouse=True)
def no_output_artifacts_in_repository():
    """A run directory is provenance, not test scratch.

    A test that leaks one into the repository's `runs/` makes a development
    fixture indistinguishable from a real certificate run in the inventory, and a
    leak into `results/` or `figures/` would touch historical evidence.
    """
    def listing():
        return {name: sorted(p.name for p in (REPOSITORY / name).iterdir())
                for name in OUTPUT_ROOTS if (REPOSITORY / name).is_dir()}

    before = listing()
    yield
    after = listing()
    assert after == before, f"test wrote artifacts into the repository: {before} -> {after}"
