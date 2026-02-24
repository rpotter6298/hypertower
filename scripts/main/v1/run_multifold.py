#!/usr/bin/env python3
"""CLI wrapper that delegates to classes.frontend.Multifold."""

from pathlib import Path
import sys

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.frontend import Multifold


def run_cli(cli_args=None):
    parser = Multifold.build_parser()
    args = parser.parse_args(cli_args)
    runner = Multifold(args)
    runner.run()


def main():
    run_cli()


if __name__ == "__main__":
    main()
