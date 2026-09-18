#!/usr/bin/env python3
"""Dependency-free runner for EAGOR's no-fixture unit tests.

The audited ``habitat`` conda environment does not currently contain pytest.
This runner executes the same plain assertion functions on CPU without changing
the environment.  They remain pytest-discoverable when pytest is installed.
"""

from __future__ import annotations

import importlib


TEST_MODULES = [
    "eagor_repro.tests.test_spherical_math",
    "eagor_repro.tests.test_policies",
    "eagor_repro.tests.test_synthetic_scenarios",
    "eagor_repro.tests.test_perception_controller",
    "eagor_repro.tests.test_evaluator_helpers",
    "eagor_repro.tests.test_navigation_attribution",
    "eagor_repro.tests.test_online_navigation",
    "eagor_repro.tests.test_stop_diagnosis",
]


def main() -> None:
    tests = []
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        tests.extend(
            (f"{module_name}.{name}", getattr(module, name))
            for name in dir(module)
            if name.startswith("test_")
        )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"PASS all {len(tests)} EAGOR tests")


if __name__ == "__main__":
    main()
