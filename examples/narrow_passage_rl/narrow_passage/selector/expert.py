"""Rule teacher used only to initialize the learned selector."""

from __future__ import annotations

import numpy as np

from .contract import Mode, coerce_memory_context, coerce_observation


def training_rule_mode(
    observation,
    memory_context=None,
    *,
    feasible_label: bool | None,
) -> Mode:
    """Return a four-mode supervision label.

    `feasible_label` is simulator truth and is intentionally required as a
    keyword-only training label.  It is never appended to the actor input.
    This teacher is distinct from the deployment Rule-19D baseline.
    """

    obs = coerce_observation(observation)
    memory = coerce_memory_context(memory_context)
    heading = abs(float(obs[10]))
    lateral = abs(float(obs[11]))
    stuck = float(obs[15])
    collision = float(obs[16])
    supported_failures = float(memory[1])
    recurrence_risk = float(memory[3])

    if feasible_label is False:
        return Mode.REJECT
    if supported_failures >= 3.0 and recurrence_risk >= 0.8:
        return Mode.REJECT
    if collision > 0.5 or stuck >= 0.7 or heading > 0.60 or lateral > 0.18:
        return Mode.RECOVER
    estimated_margin = float(obs[8]) - 0.42
    if heading > 0.20 or lateral > 0.10 or estimated_margin < 0.08:
        return Mode.EXPLORE
    return Mode.COMMIT
