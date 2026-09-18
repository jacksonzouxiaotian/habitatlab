import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narrow_passage.selector.visual_model import (  # noqa: E402
    ACTION_OUTCOME_DIM,
    NUM_MODES,
    VisualFourModeGRUPolicy,
)


def inputs(time_steps=3, batch_size=2, size=64):
    return {
        "depth": torch.rand(time_steps, batch_size, size, size, 1),
        "rgb": torch.randint(
            0, 256, (time_steps, batch_size, size, size, 3), dtype=torch.uint8
        ),
        "pointgoal": torch.rand(time_steps, batch_size, 2),
        "action_outcome": torch.zeros(
            time_steps, batch_size, ACTION_OUTCOME_DIM
        ),
        "episode_starts": torch.zeros(time_steps, batch_size),
    }


@pytest.mark.parametrize("input_type", ["depth", "rgbd"])
def test_visual_policy_sequence_contract(input_type):
    batch = inputs()
    policy = VisualFourModeGRUPolicy(
        input_type=input_type,
        hidden_size=64,
        memory_dim=4,
        privileged_critic=True,
        baseplanes=16,
    )
    logits, values, hidden = policy.forward_sequence(
        batch["depth"],
        batch["pointgoal"],
        batch["action_outcome"],
        policy.initial_hidden(2),
        rgb=batch["rgb"] if input_type == "rgbd" else None,
        memory_context=torch.zeros(3, 2, 4),
        episode_starts=batch["episode_starts"],
        privileged_19d=torch.zeros(3, 2, 19),
    )
    assert logits.shape == (3, 2, NUM_MODES)
    assert values.shape == (3, 2)
    assert hidden.shape == (1, 2, 64)
    assert policy.uses_handcrafted_actor_features is False
    assert policy.actor.in_features == 68


def test_single_step_contract_and_distribution():
    batch = inputs(time_steps=1, batch_size=2)
    policy = VisualFourModeGRUPolicy(hidden_size=32, baseplanes=16)
    dist, values, hidden = policy.distribution_and_value(
        batch["depth"][0],
        batch["pointgoal"][0],
        batch["action_outcome"][0],
        policy.initial_hidden(2),
        episode_starts=torch.ones(2),
    )
    assert dist.logits.shape == (1, 2, NUM_MODES)
    assert values.shape == (1, 2)
    assert hidden.shape == (1, 2, 32)


def test_actor_refuses_19d_when_privileged_critic_is_disabled():
    batch = inputs(time_steps=1, batch_size=1)
    policy = VisualFourModeGRUPolicy(hidden_size=32, baseplanes=16)
    with pytest.raises(ValueError, match="privileged_19d is disabled"):
        policy.forward_sequence(
            batch["depth"],
            batch["pointgoal"],
            batch["action_outcome"],
            policy.initial_hidden(1),
            privileged_19d=torch.zeros(1, 1, 19),
        )


def test_rgbd_policy_requires_rgb_and_checks_shapes():
    batch = inputs(time_steps=1, batch_size=1)
    policy = VisualFourModeGRUPolicy(
        input_type="rgbd", hidden_size=32, baseplanes=16
    )
    with pytest.raises(ValueError, match="requires rgb"):
        policy.forward_sequence(
            batch["depth"],
            batch["pointgoal"],
            batch["action_outcome"],
            policy.initial_hidden(1),
        )

