import numpy as np

from habitat.tasks.narrow_passage.geometry import (
    NarrowPassageState,
    depth_to_metric_units,
    depth_to_passage_features,
)


def test_normalized_habitat_depth_is_converted_to_metres():
    depth = np.full((64, 64, 1), 0.10, dtype=np.float32)
    features = depth_to_passage_features(
        depth,
        state=NarrowPassageState(robot_radius=0.18),
        depth_is_normalized=True,
        depth_min=0.0,
        depth_max=10.0,
    )

    assert np.isclose(features[6], 1.0)
    assert np.isclose(features[7], 1.0)
    assert np.isclose(features[8], 2.0)
    assert np.isclose(features[9], 0.82)


def test_metric_depth_callers_keep_the_historical_api_and_units():
    depth = np.full((64, 64), 1.0, dtype=np.float32)
    features = depth_to_passage_features(
        depth, state=NarrowPassageState(robot_radius=0.18)
    )

    assert np.isclose(features[9], 0.82)


def test_invalid_depth_range_fails_fast():
    with np.testing.assert_raises(ValueError):
        depth_to_metric_units(
            np.ones((2, 2), dtype=np.float32),
            normalized=True,
            min_depth=1.0,
            max_depth=1.0,
        )
