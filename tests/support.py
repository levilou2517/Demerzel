"""Shared test fixtures.

Tests use ``unittest`` (stdlib) so the suite runs with no third-party test
runner installed. The layout is pytest-compatible: every ``test_*.py`` module
uses ``unittest.TestCase`` classes named ``Test*``, which pytest collects
natively, so ``pytest`` and ``python -m unittest discover`` both work.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

# Make the repository importable without installation.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from demerzel.config import Config  # noqa: E402
from demerzel.engine import Engine  # noqa: E402
from demerzel.reference import (  # noqa: E402
    DictGraph,
    HashingEmbeddingProvider,
    InMemoryEvidenceStore,
    InMemoryStorage,
    MockLLMProvider,
)


class DemerzelTestCase(unittest.TestCase):
    """Base class wiring an in-memory engine for a test."""

    def make_engine(self, config: Config | None = None, llm=None) -> Engine:
        """Build an engine over fresh in-memory backends."""
        self.storage = InMemoryStorage()
        self.evidence = InMemoryEvidenceStore()
        self.graph = DictGraph()
        self.embedder = HashingEmbeddingProvider(256)
        self.llm = llm if llm is not None else MockLLMProvider()
        self.config = config or Config()
        return Engine(
            self.storage,
            self.graph,
            self.embedder,
            self.evidence,
            self.llm,
            self.config,
        )

    def temp_path(self, name: str = "demerzel_test.db") -> str:
        """A path inside a per-test temporary directory."""
        if not hasattr(self, "_tmpdir"):
            self._tmpdir = tempfile.mkdtemp(prefix="demerzel-test-")
        return os.path.join(self._tmpdir, name)

    def tearDown(self) -> None:
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir and os.path.isdir(tmpdir):
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
