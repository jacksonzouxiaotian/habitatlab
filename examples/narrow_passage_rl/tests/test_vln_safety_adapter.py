import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narrow_passage.models.policy import DecisionMode
from narrow_passage.models.vln_safety_adapter import (
    VLNAdapterObservation,
    VLNSafetyAdapter,
    parse_trusted_control_directive,
)


def test_trusted_stop_override_has_priority():
    adapter = VLNSafetyAdapter()
    decision = adapter.adapt(
        "move_forward",
        75.0,
        VLNAdapterObservation(
            instruction=(
                "Walk through the room. Instruction update: the task is "
                "complete; stop immediately."
            ),
            trusted_control_directive=True,
        ),
    )
    assert decision.adapted_action == "stop"
    assert decision.terminal_stop
    assert decision.mode == DecisionMode.REJECT


def test_route_instruction_is_not_parsed_without_trust():
    adapter = VLNSafetyAdapter()
    decision = adapter.adapt(
        "move_forward",
        50.0,
        VLNAdapterObservation(
            instruction="Turn left after the sofa and stop by the table.",
            trusted_control_directive=False,
        ),
    )
    assert decision.adapted_action == "move_forward"
    assert decision.mode == DecisionMode.COMMIT
    assert not decision.overridden


def test_low_clearance_rejects_forward_proposal():
    adapter = VLNSafetyAdapter()
    decision = adapter.adapt(
        "move_forward",
        50.0,
        VLNAdapterObservation(
            instruction="Proceed through the doorway.",
            p_feas=0.05,
            risk=0.95,
            min_clearance_m=0.02,
            body_margin_m=-0.01,
        ),
    )
    assert decision.adapted_action == "stop"
    assert decision.mode == DecisionMode.REJECT
    assert decision.intervention == "safety"


def test_stuck_state_requests_recovery():
    adapter = VLNSafetyAdapter()
    decision = adapter.adapt(
        "move_forward",
        25.0,
        VLNAdapterObservation(
            instruction="Continue toward the goal.",
            stuck_score=0.9,
        ),
    )
    assert decision.adapted_action == "stop"
    assert decision.mode == DecisionMode.RECOVER


def test_uncertain_forward_action_is_capped():
    adapter = VLNSafetyAdapter()
    decision = adapter.adapt(
        "move_forward",
        75.0,
        VLNAdapterObservation(
            instruction="Continue.",
            p_feas=0.55,
            risk=0.45,
        ),
    )
    assert decision.mode == DecisionMode.EXPLORE
    assert decision.adapted_value == 20.0


def test_trusted_direction_is_parsed_but_route_is_not_implicitly_parsed():
    directive = parse_trusted_control_directive("Turn right 45 degrees.")
    assert directive is not None
    assert directive.action == "turn_right"
    assert directive.value == 45.0
