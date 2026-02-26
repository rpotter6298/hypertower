#!/usr/bin/env python3
"""CLI wrapper for the V2 hypertower pipeline using V2HyperTower directly."""

from __future__ import annotations

from pathlib import Path
import sys

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.v2_hypertower import V2HyperTower


def run_cli(cli_args=None):
    parser = V2HyperTower.build_parser()
    args = parser.parse_args(cli_args)
    V2HyperTower(args).run()


def main():
    run_cli()


if __name__ == "__main__":
    main()
