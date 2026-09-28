#!/usr/bin/env python3
"""Run the whole Demerzel test suite.

Works with the standard library alone (``unittest``), so it runs in
environments where ``pytest`` cannot be installed. When ``pytest`` is present,
prefer ``pytest`` — it collects the same ``unittest.TestCase`` classes.

Usage::

    python tests/run_tests.py            # quiet
    python tests/run_tests.py -v         # verbose
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    verbosity = 2 if "-v" in sys.argv else 1
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=HERE, pattern="test_*.py", top_level_dir=HERE)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())