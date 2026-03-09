from __future__ import annotations

import sys

from omegaconf import OmegaConf
from rich.console import Console

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from kuka_planner_simple.pipelines.experiment_runtime import load_cfg


def main():
    console = Console()
    cfg = load_cfg(sys.argv[1:])
    console.print("[bold]Resolved config[/bold]")
    console.print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
