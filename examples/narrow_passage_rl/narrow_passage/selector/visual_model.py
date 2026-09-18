"""End-to-end visual recurrent policy for the four DEGNAV modes.

The actor consumes raw Habitat images, PointGoal, compact action/outcome state,
and an optional cross-episode memory vector.  The hand-crafted 19-D geometry
features are deliberately absent from the actor interface.  They may be passed
to the critic as privileged training-only information, which makes that use
explicit and prevents accidental feature leakage at deployment.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.distributions import Categorical

from habitat_baselines.rl.ddppo.policy import resnet

from .contract import MEMORY_DIM, OBSERVATION_DIM


InputType = Literal["depth", "rgbd"]
POINTGOAL_DIM = 2
# Previous mode as one-hot (4), previous collision, previous stuck, and previous
# normalized progress.  Recurrent visual history remains in the GRU state.
ACTION_OUTCOME_DIM = 7
NUM_MODES = 4


class VisualFourModeGRUPolicy(nn.Module):
    """Raw Depth/RGB-D + PointGoal + history -> four DEGNAV modes.

    Tensor layout follows Habitat observations: images are ``[T,B,H,W,C]`` or
    ``[B,H,W,C]``.  ``pointgoal`` and ``action_outcome`` are ``[T,B,D]`` or
    ``[B,D]``.  Depth is expected in Habitat's normalized ``[0,1]`` range and
    RGB in uint8 ``[0,255]`` or floating-point ``[0,1]``.
    """

    def __init__(
        self,
        input_type: InputType = "depth",
        hidden_size: int = 256,
        memory_dim: int = 0,
        privileged_critic: bool = False,
        baseplanes: int = 32,
    ) -> None:
        super().__init__()
        if input_type not in ("depth", "rgbd"):
            raise ValueError("input_type must be 'depth' or 'rgbd'")
        if memory_dim not in (0, MEMORY_DIM):
            raise ValueError(f"memory_dim must be 0 or {MEMORY_DIM}")

        self.input_type = input_type
        self.hidden_size = int(hidden_size)
        self.memory_dim = int(memory_dim)
        self.privileged_dim = OBSERVATION_DIM if privileged_critic else 0
        in_channels = 1 if input_type == "depth" else 4

        # This is the same GroupNorm ResNet-18 family used by Habitat's released
        # PointNav PPO, with adaptive pooling so the contract is resolution-safe.
        self.visual_backbone = resnet.resnet18(
            in_channels=in_channels,
            base_planes=int(baseplanes),
            # Match the released PointNav encoder's GroupNorm grouping so the
            # copied backbone weights retain their original semantics.
            ngroups=max(1, int(baseplanes) // 2),
        )
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.visual_backbone.final_channels, hidden_size),
            nn.ELU(),
        )
        self.pointgoal_encoder = nn.Sequential(
            nn.Linear(POINTGOAL_DIM, 32),
            nn.ELU(),
        )
        self.action_outcome_encoder = nn.Sequential(
            nn.Linear(ACTION_OUTCOME_DIM, 32),
            nn.ELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ELU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.actor = nn.Linear(hidden_size + memory_dim, NUM_MODES)
        self.critic = nn.Linear(hidden_size + self.privileged_dim, 1)
        self._init_parameters()

    @property
    def uses_handcrafted_actor_features(self) -> bool:
        """Machine-checkable declaration used by experiment audits."""

        return False

    def _init_parameters(self) -> None:
        for module in (
            self.visual_projection,
            self.pointgoal_encoder,
            self.action_outcome_encoder,
            self.fusion,
            self.actor,
            self.critic,
        ):
            for layer in module.modules() if isinstance(module, nn.Sequential) else (module,):
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def initial_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    @staticmethod
    def _as_sequence(value: torch.Tensor, feature_ndim: int) -> torch.Tensor:
        if value.ndim == feature_ndim + 1:
            return value.unsqueeze(0)
        if value.ndim != feature_ndim + 2:
            raise ValueError(
                f"expected rank {feature_ndim + 1} or {feature_ndim + 2}, "
                f"got shape {tuple(value.shape)}"
            )
        return value

    def _visual_input(
        self, depth: torch.Tensor, rgb: torch.Tensor | None
    ) -> tuple[torch.Tensor, int, int]:
        depth = self._as_sequence(depth, feature_ndim=3)
        if depth.shape[-1] != 1:
            raise ValueError(f"depth must end in one channel, got {depth.shape}")
        if not torch.isfinite(depth).all():
            raise ValueError("depth contains NaN or Inf")
        depth = depth.float().clamp(0.0, 1.0)
        time_steps, batch_size = depth.shape[:2]
        streams = [depth]

        if self.input_type == "rgbd":
            if rgb is None:
                raise ValueError("rgbd policy requires rgb")
            rgb = self._as_sequence(rgb, feature_ndim=3)
            if rgb.shape[:4] != depth.shape[:4] or rgb.shape[-1] != 3:
                raise ValueError(
                    f"rgb/depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}"
                )
            if not torch.isfinite(rgb.float()).all():
                raise ValueError("rgb contains NaN or Inf")
            rgb = rgb.float()
            if rgb.detach().amax().item() > 1.0:
                rgb = rgb / 255.0
            streams.insert(0, rgb.clamp(0.0, 1.0))
        elif rgb is not None:
            raise ValueError("depth policy must not receive an rgb tensor")

        visual = torch.cat(streams, dim=-1)
        visual = visual.reshape(
            time_steps * batch_size, *visual.shape[2:]
        ).permute(0, 3, 1, 2)
        return visual, time_steps, batch_size

    def forward_sequence(
        self,
        depth: torch.Tensor,
        pointgoal: torch.Tensor,
        action_outcome: torch.Tensor,
        hidden: torch.Tensor,
        *,
        rgb: torch.Tensor | None = None,
        memory_context: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        privileged_19d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual, time_steps, batch_size = self._visual_input(depth, rgb)
        pointgoal = self._as_sequence(pointgoal, feature_ndim=1).float()
        action_outcome = self._as_sequence(action_outcome, feature_ndim=1).float()
        expected_prefix = (time_steps, batch_size)
        if pointgoal.shape != (*expected_prefix, POINTGOAL_DIM):
            raise ValueError(f"invalid pointgoal shape {tuple(pointgoal.shape)}")
        if action_outcome.shape != (*expected_prefix, ACTION_OUTCOME_DIM):
            raise ValueError(
                f"invalid action_outcome shape {tuple(action_outcome.shape)}"
            )
        if not torch.isfinite(pointgoal).all() or not torch.isfinite(action_outcome).all():
            raise ValueError("non-visual actor input contains NaN or Inf")

        visual_features = self.visual_projection(
            self.visual_pool(self.visual_backbone(visual))
        ).reshape(time_steps, batch_size, self.hidden_size)
        goal_features = self.pointgoal_encoder(pointgoal)
        action_features = self.action_outcome_encoder(action_outcome)
        fused = self.fusion(
            torch.cat([visual_features, goal_features, action_features], dim=-1)
        )

        if episode_starts is not None:
            episode_starts = self._as_sequence(episode_starts, feature_ndim=0)
            if episode_starts.shape != expected_prefix:
                raise ValueError(
                    f"invalid episode_starts shape {tuple(episode_starts.shape)}"
                )

        outputs = []
        state = hidden
        for step in range(time_steps):
            if episode_starts is not None:
                reset = episode_starts[step].reshape(1, batch_size, 1).to(fused.dtype)
                state = state * (1.0 - reset)
            output, state = self.gru(fused[step : step + 1], state)
            outputs.append(output)
        recurrent = torch.cat(outputs, dim=0)

        if self.memory_dim:
            if memory_context is None:
                raise ValueError("memory-conditioned policy requires memory_context")
            memory_context = self._as_sequence(memory_context, feature_ndim=1).float()
            if memory_context.shape != (*expected_prefix, self.memory_dim):
                raise ValueError(
                    f"invalid memory_context shape {tuple(memory_context.shape)}"
                )
            actor_features = torch.cat([recurrent, memory_context], dim=-1)
        else:
            if memory_context is not None:
                raise ValueError("memory-free policy must not receive memory_context")
            actor_features = recurrent
        logits = self.actor(actor_features)

        if self.privileged_dim:
            if privileged_19d is None:
                raise ValueError("privileged critic requires privileged_19d")
            privileged_19d = self._as_sequence(privileged_19d, feature_ndim=1).float()
            if privileged_19d.shape != (*expected_prefix, OBSERVATION_DIM):
                raise ValueError(
                    f"invalid privileged_19d shape {tuple(privileged_19d.shape)}"
                )
            critic_features = torch.cat([recurrent, privileged_19d], dim=-1)
        else:
            if privileged_19d is not None:
                raise ValueError(
                    "privileged_19d is disabled; refusing accidental actor/critic leakage"
                )
            critic_features = recurrent
        values = self.critic(critic_features).squeeze(-1)
        return logits, values, state

    def distribution_and_value(self, *args, **kwargs):
        logits, values, hidden = self.forward_sequence(*args, **kwargs)
        return Categorical(logits=logits), values, hidden
