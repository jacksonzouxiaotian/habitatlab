#!/usr/bin/env python3

from typing import Dict, Optional

import torch
from gym import spaces
from torch import nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo.policy import Net, NetPolicy


class NarrowPassageNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        feature_key: str = "narrow_passage_features",
        memory_key: str = "narrow_passage_memory",
    ) -> None:
        super().__init__()
        if feature_key not in observation_space.spaces:
            raise KeyError(
                f"{feature_key} is required by NarrowPassagePolicy. "
                f"Available observations: {list(observation_space.spaces.keys())}"
            )

        self._feature_key = feature_key
        self._memory_key = memory_key if memory_key in observation_space.spaces else None
        self._hidden_size = hidden_size
        feature_dim = observation_space.spaces[feature_key].shape[0]
        if self._memory_key is not None:
            feature_dim += observation_space.spaces[self._memory_key].shape[0]

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.state_encoder = build_rnn_state_encoder(hidden_size, hidden_size)
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

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ):
        features = [observations[self._feature_key].float()]
        if self._memory_key is not None:
            features.append(observations[self._memory_key].float())
        x = torch.cat(features, dim=-1)
        x = self.encoder(x)
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
    ) -> None:
        super().__init__(
            NarrowPassageNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
            ),
            action_space=action_space,
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
        )
