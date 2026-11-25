"""Protect dependency boundaries and import order across runtime packages."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("first", ["rover.cache.kivi", "rover.storage.files", "rover.models.qwen3.text"])
def test_packages_import_in_any_order(first):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {first}; from rover.engine import Engine; "
            "from rover.runtime.state import GenerationState; "
            "from rover.scheduling.prefill import PrefillScheduler",
        ],
        check=True,
    )


def test_caches_do_not_depend_on_model_implementations():
    import rover.cache

    for path in Path(rover.cache.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("rover.models.qwen3"), path
                assert node.module != "rover.runtime.runner", path


def test_storage_primitives_do_not_import_runtime_state():
    import rover.storage.files

    tree = ast.parse(Path(rover.storage.files.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "rover.runtime.state" not in imports
    assert "rover.runtime.runner" not in imports
