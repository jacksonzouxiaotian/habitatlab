#!/usr/bin/env python3
"""Evaluate narrow-passage methods.

This wrapper keeps a stable paper-facing command while delegating to the tested
evaluators.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fsm", "sb3", "apf_gap"], default="fsm")
    args, rest = parser.parse_known_args()

    if args.method == "fsm":
        from examples.narrow_passage_rl.eval_habitat_geometry_fsm import main as run
    elif args.method == "sb3":
        from examples.narrow_passage_rl.eval_habitat_sb3 import main as run
    else:
        from examples.narrow_passage_rl.eval_habitat_apf_gap import main as run

    # Delegated scripts parse sys.argv; forward only the remaining arguments.
    sys.argv = [sys.argv[0], *rest]
    run()


if __name__ == "__main__":
    main()
