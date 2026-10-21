"""Shared helpers for the test modules.

Every test module runs both under pytest (``pytest -q``) and on its own
(``python3 tests/test_x.py``), without any hardware, camera or MediaPipe.
Importing this module puts the project root and ``vendor/`` on ``sys.path``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telehand.bootstrap import bootstrap  # noqa: E402

bootstrap()


def check(cond: bool, label: str) -> None:
    """Print a PASS/FAIL line, then assert so pytest reports the label."""
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


def run_standalone(namespace: dict) -> int:
    """Run every ``test_*`` function in ``namespace`` in definition order."""
    tests = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        print(test.__name__[len("test_") :].replace("_", " "))
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAILED: {exc}")
        except Exception:
            failures += 1
            traceback.print_exc()
    print()
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0
