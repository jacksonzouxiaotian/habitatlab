import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.logging_schema import (  # noqa: E402
    CANONICAL_CSV_FIELDS,
    fieldnames_for_rows,
    finalize_episode_row,
    mode_count_ratio_fields,
)


def test_mode_counts_and_ratios():
    row = mode_count_ratio_fields(
        {"commit": 2, "explore": 1, "recover": 1, "reject": 0},
        mode_interface_available=True,
    )
    assert row["total_mode_count"] == 4
    assert row["commit_ratio"] == 0.5
    assert row["explore_ratio"] == 0.25
    assert row["recover_ratio"] == 0.25
    assert row["reject_ratio"] == 0.0


def test_zero_total_count_no_divide_by_zero():
    row = mode_count_ratio_fields({}, mode_interface_available=True)
    assert row["total_mode_count"] == 0
    assert row["commit_ratio"] == 0.0
    assert row["explore_ratio"] == 0.0
    assert row["recover_ratio"] == 0.0
    assert row["reject_ratio"] == 0.0


def test_no_belief_baseline_outputs_nan():
    row = finalize_episode_row(
        {"episode_id": "ep0", "method": "apf", "success": 1.0},
        belief_available=False,
        mode_interface_available=False,
        final_mode="apf",
    )
    assert row["belief_available"] is False
    assert math.isnan(row["d_hat"])
    assert math.isnan(row["p_feas"])
    assert math.isnan(row["risk_mean"])


def test_no_mode_interface_is_not_zero_recover_usage():
    row = finalize_episode_row(
        {"episode_id": "ep0", "method": "direct_velocity"},
        belief_available=False,
        mode_interface_available=False,
        final_mode="direct_velocity",
    )
    assert row["mode_interface_available"] is False
    assert row["final_mode"] == "direct_velocity"
    assert math.isnan(row["recover_count"])
    assert math.isnan(row["recover_ratio"])


def test_csv_header_stable():
    row = finalize_episode_row({"episode_id": "ep0", "custom": 1})
    fields = fieldnames_for_rows([row])
    assert fields[: len(CANONICAL_CSV_FIELDS)] == CANONICAL_CSV_FIELDS
    assert fields[-1] == "custom"
