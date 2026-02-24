#!/usr/bin/env python3
"""CLI wrapper that delegates to classes.frontend.Multifold with V2 loaders."""

from __future__ import annotations

from pathlib import Path
import sys

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import classes.frontend as frontend
from classes.v2.v2_hypertower import V2HyperTower


def run_cli(cli_args=None):
    parser = frontend.Multifold.build_parser()
    parser.set_defaults(warmup_tower_epochs=None, warmup_fused_epochs=None)
    parser.add_argument(
        "--sample-mode",
        choices=["eye", "patient"],
        default="eye",
        help="Build samples per eye (row-level) or per patient (multi-slot).",
    )
    args = parser.parse_args(cli_args)

    # Monkeypatch the HyperTower class used inside Multifold.
    frontend.HyperTower = V2HyperTower

    runner = frontend.Multifold(args)
    runner.run()


def main():
    run_cli()


if __name__ == "__main__":
    main()
