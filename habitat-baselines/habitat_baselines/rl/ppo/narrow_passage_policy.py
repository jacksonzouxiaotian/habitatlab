#!/usr/bin/env python3

from typing import Dict, Optional

import torch
from gym import spaces
from torch import nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo.policy import Net, NetPolicy


GEOMETRY_SLICE = slice(0, 10)
GOAL_SLICE = slice(10, 13)
PROPRIO_INDICES = (13, 14, 15, 16, 17, 18)
CLEARANCE_INDICES = (0, 2, 3, 5, 6, 7, 8, 9)
RISK_INDICES = (15, 16)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(inplace=True),
    )


class _IdentityStateEncoder(nn.Module):
    num_recurrent_layers = 0

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.recurrent_hidden_size = hidden_size

    def forward(
        self,
        x,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info=None,
    ):
        return x, rnn_hidden_states


class NarrowPassageNet(Net):
    """Low-dimensional policy net for NarrowPassageNav-v0.

    Habitat-Baselines' NetPolicy owns the Gaussian actor head and critic head.
    This net only turns the 19-D interpretable narrow-passage observation into a
    compact embedding suitable for those heads.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        feature_key: str = "narrow_passage_features",
        memory_key: str = "narrow_passage_memory",
        use_recurrent: bool = True,
        ablation: str = "full",
        use_memory_features: bool = False,
        use_geometry_features: bool = True,
        use_clearance_features: bool = True,
        use_risk_features: bool = True,
    ) -> None:
        super().__init__()
        if feature_key not in observation_space.spaces:
            raise KeyError(
                f"{feature_key} is required by NarrowPassagePolicy. "
                f"Available observations: {list(observation_space.spaces.keys())}"
            )

        feature_dim = observation_space.spaces[feature_key].shape[0]
        if feature_dim != 19:
            raise ValueError(
                f"NarrowPassagePolicy expects 19-D features, got {feature_dim}"
            )
        if ablation not in {"full", "without_geometry", "without_clearance", "without_risk"}:
            raise ValueError(
                "ablation must be one of full, without_geometry, "
                "without_clearance, without_risk"
            )

        self._feature_key = feature_key
        self._memory_key = (
            memory_key
            if use_memory_features and memory_key in observation_space.spaces
            else None
        )
        self._hidden_size = hidden_size
        self._use_recurrent = use_recurrent
        self._ablation = ablation
        self._use_geometry_features = use_geometry_features
        self._use_clearance_features = use_clearance_features
        self._use_risk_features = use_risk_features

        branch_size = max(32, hidden_size // 4)
        self.geometry_encoder = _mlp(10, branch_size, branch_size)
        self.goal_encoder = _mlp(3, branch_size, branch_size)
        self.proprio_encoder = _mlp(len(PROPRIO_INDICES), branch_size, branch_size)

        fusion_input = branch_size * 3
        if self._memory_key is not None:
            memory_dim = observation_space.spaces[self._memory_key].shape[0]
            self.memory_encoder = _mlp(memory_dim, branch_size, branch_size)
            fusion_input += branch_size
        else:
            self.memory_encoder = None

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        if use_recurrent:
            self.state_encoder = build_rnn_state_encoder(hidden_size, hidden_size)
        else:
            self.state_encoder = _IdentityStateEncoder(hidden_size)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    @property
    def visual_encoder(self) -> Optional[nn.Module]:
        return None

    def _apply_ablation(self, features: torch.Tensor) -> torch.Tensor:
        features = features.clone()
        if self._ablation == "without_geometry" or not self._use_geometry_features:
            features[:, GEOMETRY_SLICE] = 0.0
        if self._ablation == "without_clearance" or not self._use_clearance_features:
            features[:, list(CLEARANCE_INDICES)] = 0.0
        if self._ablation == "without_risk" or not self._use_risk_features:
            features[:, list(RISK_INDICES)] = 0.0
        return features

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ):
        features = observations[self._feature_key].float()
        features = self._apply_ablation(features)

        geometry = self.geometry_encoder(features[:, GEOMETRY_SLICE])
        goal = self.goal_encoder(features[:, GOAL_SLICE])
        proprio = self.proprio_encoder(features[:, list(PROPRIO_INDICES)])
        encoded = [geometry, goal, proprio]
        if self._memory_key is not None and self.memory_encoder is not None:
            encoded.append(self.memory_encoder(observations[self._memory_key].float()))

        x = torch.cat(encoded, dim=-1)
        x = self.fusion_mlp(x)
        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states, masks, rnn_build_seq_info
        )
        return x, rnn_hidden_states, {"narrow_passage_embedding": x}


@baseline_registry.register_policy
class NarrowPassagePolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 256,
        policy_config=None,
        aux_loss_config=None,
        feature_key: str = "narrow_passage_features",
        memory_key: str = "narrow_passage_memory",
        use_recurrent: bool = True,
        ablation: str = "full",
        use_memory_features: bool = False,
        use_geometry_features: bool = True,
        use_clearance_features: bool = True,
        use_risk_features: bool = True,
    ) -> None:
        super().__init__(
            NarrowPassageNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                feature_key=feature_key,
                memory_key=memory_key,
                use_recurrent=use_recurrent,
                ablation=ablation,
                use_memory_features=use_memory_features,
                use_geometry_features=use_geometry_features,
                use_clearance_features=use_clearance_features,
                use_risk_features=use_risk_features,
            ),
            action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        policy_cfg = config.habitat_baselines.rl.policy.main_agent
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            policy_config=policy_cfg,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            feature_key=getattr(policy_cfg, "feature_key", "narrow_passage_features"),
            memory_key=getattr(policy_cfg, "memory_key", "narrow_passage_memory"),
            use_recurrent=bool(getattr(policy_cfg, "use_recurrent", True)),
            ablation=getattr(policy_cfg, "ablation", "full"),
            use_memory_features=bool(getattr(policy_cfg, "use_memory_features", False)),
            use_geometry_features=bool(
                getattr(policy_cfg, "use_geometry_features", True)
            ),
            use_clearance_features=bool(
                getattr(policy_cfg, "use_clearance_features", True)
            ),
            use_risk_features=bool(getattr(policy_cfg, "use_risk_features", True)),
        )
