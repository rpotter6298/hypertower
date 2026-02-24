#!/usr/bin/env python3
"""CLI wrapper for the V2 three-mode comparison (classic/ensemble/bilateral)."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.v2_hypertower import V2ModeComparator


def main():
    V2ModeComparator.run()


if __name__ == "__main__":
    main()
