#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from eagor_repro.config import load_config
from eagor_repro.evaluation.habitat_evaluator import (
    evaluate_objectnav,
    habitat_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="EAGOR Habitat ObjectNav evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/eagor/base.yaml"))
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--policy", default="eagor")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.overlay:
        config = OmegaConf.merge(config, OmegaConf.load(args.overlay))
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    OmegaConf.resolve(config)
    if args.preflight:
        print(json.dumps(habitat_preflight(config), indent=2))
        return
    print(json.dumps(evaluate_objectnav(config, args.policy), indent=2))


if __name__ == "__main__":
    main()
