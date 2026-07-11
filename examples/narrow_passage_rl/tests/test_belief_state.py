import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narrow_passage.models.belief_state import BeliefState, BeliefStateConfig


def _obs(passage_width=0.55):
    obs = np.zeros(19, dtype=np.float32)
    obs[6] = passage_width / 2.0
    obs[7] = passage_width / 2.0
    obs[8] = passage_width
    obs[9] = passage_width / 2.0 - 0.18
    obs[10] = 0.10
    obs[11] = -0.03
    obs[15] = 0.20
    obs[16] = 0.0
    return obs


def test_p_feas_increases_when_d_hat_increases():
    cfg = BeliefStateConfig(sigma_d=0.05, sigma_w=0.05)
    low = BeliefState.from_obs(_obs(0.42), w_req_cons=0.50, cfg=cfg)
    high = BeliefState.from_obs(_obs(0.62), w_req_cons=0.50, cfg=cfg)
    assert high.p_feas > low.p_feas


def test_p_feas_decreases_when_required_width_increases():
    obs = _obs(0.56)
    easier = BeliefState.from_obs(obs, w_req_cons=0.46)
    harder = BeliefState.from_obs(obs, w_req_cons=0.66)
    assert harder.p_feas < easier.p_feas


def test_as_array_length_matches_feature_names():
    belief = BeliefState.from_obs(_obs())
    assert belief.as_array().shape == (len(BeliefState.feature_names()),)


def test_invalid_obs_dimension_raises_value_error():
    with pytest.raises(ValueError, match="Expected 19-D"):
        BeliefState.from_obs(np.zeros(18, dtype=np.float32))


def test_all_returned_features_are_finite_float32():
    belief = BeliefState.from_obs(
        {"narrow_passage_features": _obs()},
        memory_risk=0.25,
        w_req_prior=0.36,
        w_req_cons=0.44,
    )
    arr = belief.as_array()
    assert arr.dtype == np.float32
    assert np.all(np.isfinite(arr))
