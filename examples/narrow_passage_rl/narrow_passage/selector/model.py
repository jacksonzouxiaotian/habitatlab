"""Lightweight recurrent actor-critic with a strict 19-D actor input."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .contract import MEMORY_DIM, OBSERVATION_DIM


class RunningMeanStd(nn.Module):
    """Serializable running normalization using the parallel-variance update."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, *self.mean.shape).float()
        if values.numel() == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def normalize(self, values: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        normalized = (values - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normalized, -clip, clip)


class FourModeGRUPolicy(nn.Module):
    """19→128→128→GRU actor/critic; memory is a separate optional input."""

    def __init__(
        self,
        hidden_size: int = 128,
        memory_dim: int = 0,
        privileged_dim: int = 0,
    ) -> None:
        super().__init__()
        if memory_dim not in (0, MEMORY_DIM):
            raise ValueError(f"memory_dim must be 0 or {MEMORY_DIM}")
        self.obs_dim = OBSERVATION_DIM
        self.hidden_size = int(hidden_size)
        self.memory_dim = int(memory_dim)
        self.privileged_dim = int(privileged_dim)
        self.obs_rms = RunningMeanStd((OBSERVATION_DIM,))
        self.encoder = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.actor = nn.Linear(hidden_size + memory_dim, 4)
        self.critic = nn.Linear(hidden_size + privileged_dim, 1)
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def initial_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward_sequence(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        memory_context: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        privileged: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward `[T,B,19]`, resetting recurrent state at episode starts."""

        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        if episode_starts is not None and episode_starts.ndim == 1:
            episode_starts = episode_starts.unsqueeze(0)
        assert observations.shape[-1] == OBSERVATION_DIM
        assert torch.isfinite(observations).all()
        x = self.encoder(self.obs_rms.normalize(observations.float()))
        outputs = []
        state = hidden
        for step in range(x.shape[0]):
            if episode_starts is not None:
                reset = episode_starts[step].reshape(1, -1, 1).to(x.dtype)
                state = state * (1.0 - reset)
            out, state = self.gru(x[step : step + 1], state)
            outputs.append(out)
        recurrent = torch.cat(outputs, dim=0)

        if self.memory_dim:
            if memory_context is None:
                raise ValueError("memory-conditioned policy requires memory_context")
            if memory_context.ndim == 2:
                memory_context = memory_context.unsqueeze(0)
            actor_features = torch.cat([recurrent, memory_context.float()], dim=-1)
        else:
            actor_features = recurrent
        logits = self.actor(actor_features)

        if self.privileged_dim:
            if privileged is None:
                raise ValueError("privileged critic requires privileged input")
            if privileged.ndim == 2:
                privileged = privileged.unsqueeze(0)
            critic_features = torch.cat([recurrent, privileged.float()], dim=-1)
        else:
            critic_features = recurrent
        values = self.critic(critic_features).squeeze(-1)
        return logits, values, state

    def distribution_and_value(self, *args, **kwargs):
        logits, values, hidden = self.forward_sequence(*args, **kwargs)
        return Categorical(logits=logits), values, hidden

    @torch.no_grad()
    def update_normalization(self, observations: torch.Tensor) -> None:
        assert observations.shape[-1] == OBSERVATION_DIM
        assert torch.isfinite(observations).all()
        self.obs_rms.update(observations)
