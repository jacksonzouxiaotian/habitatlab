import numpy as np

from examples.narrow_passage_rl.narrow_passage.models.vln_depth_safety import (
    MOVE_FORWARD,
    OnlineDepthSafetyAdapter,
)


def test_safe_normalized_depth_commits_forward_macro():
    adapter = OnlineDepthSafetyAdapter()
    decision = adapter.decide(MOVE_FORWARD, np.ones((64, 64, 1), dtype=np.float32))
    assert decision.adapted_action == MOVE_FORWARD
    assert decision.mode == "COMMIT"
    assert decision.allow_macro_queue


def test_blocked_front_overrides_forward_with_recovery_turn():
    adapter = OnlineDepthSafetyAdapter()
    depth = np.ones((64, 64, 1), dtype=np.float32)
    depth[:, 24:40, :] = 0.02
    decision = adapter.decide(MOVE_FORWARD, depth)
    assert decision.adapted_action != MOVE_FORWARD
    assert decision.mode == "RECOVER"
    assert not decision.allow_macro_queue


def test_collision_feedback_writes_memory_and_recovers():
    adapter = OnlineDepthSafetyAdapter()
    depth = np.ones((64, 64, 1), dtype=np.float32)
    adapter.decide(MOVE_FORWARD, depth)
    adapter.observe_transition({"collisions": {"is_collision": True}})
    decision = adapter.decide(MOVE_FORWARD, depth)
    stats = adapter.episode_stats()
    assert decision.mode == "RECOVER"
    assert stats["collisions_observed"] == 1
    assert stats["memory_writes"] == 1
