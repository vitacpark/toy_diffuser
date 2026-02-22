from __future__ import annotations

import os
import sys

from rich.console import Console

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipelines.run_eval_pipeline import run_eval_from_cli


def main():
    console = Console()
    run_eval_from_cli(sys.argv[1:], console=console)


if __name__ == "__main__":
    main()
