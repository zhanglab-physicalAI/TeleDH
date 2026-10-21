"""``latency``: re-print the stage-timing summary for a CSV from an earlier run."""

from __future__ import annotations

import sys
from pathlib import Path

from .. import latency as _latency


def run(args) -> int:
    path = Path(args.csv)
    if not path.exists():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 1
    print(_latency.summarize_csv(path))
    return 0
