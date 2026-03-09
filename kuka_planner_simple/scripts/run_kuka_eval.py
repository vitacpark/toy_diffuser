from __future__ import annotations

import sys

from rich.console import Console

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kuka_planner_simple.pipelines.run_eval_pipeline import run_eval_from_cli


def main():
    console = Console()
    run_eval_from_cli(sys.argv[1:], console=console)


if __name__ == "__main__":
    main()
