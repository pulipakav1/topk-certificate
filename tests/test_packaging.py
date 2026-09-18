"""Every module the certificate pipeline imports must survive a clean install."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINTS = ["evaluation_harness.py", "analyze_results.py", "plots.py", "artifact_index.py"]


def declared_modules():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(config["tool"]["setuptools"]["py-modules"])


def imported_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def pipeline_modules():
    """First-party modules reachable from the entry points, following imports."""
    local = {p.stem for p in ROOT.glob("*.py")}
    reached, pending = set(), [ROOT / name for name in ENTRY_POINTS]
    while pending:
        path = pending.pop()
        if path.stem in reached:
            continue
        reached.add(path.stem)
        for name in imported_names(path) & local:
            pending.append(ROOT / f"{name}.py")
    return reached


def test_pipeline_imports_are_packaged():
    missing = pipeline_modules() - declared_modules()
    assert not missing, f"modules imported by the pipeline but absent from pyproject: {sorted(missing)}"


def test_every_root_module_is_packaged():
    present = {p.stem for p in ROOT.glob("*.py")}
    assert present - declared_modules() == set()
    assert declared_modules() - present == set()


def test_generated_artifacts_are_not_packaged():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools_config = config["tool"]["setuptools"]
    # py-modules alone disables package auto-discovery, which would otherwise
    # sweep in runs/, results/, figures/ and __pycache__.
    assert "packages" not in setuptools_config
    assert "package-data" not in setuptools_config
