from __future__ import annotations

from rich.console import Console
from rich.table import Table

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kuka_planner_simple.planners import supported_models


def main():
    console = Console()
    table = Table(title="Supported Model Planners")
    table.add_column("planner", style="bold")
    for name in supported_models():
        table.add_row(name)
    console.print(table)


if __name__ == "__main__":
    main()
