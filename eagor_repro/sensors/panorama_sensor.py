"""Native Habitat-Sim equirectangular sensor configuration.

The audited Habitat-Lab/Habitat-Sim 0.3.3 environment registers native RGB,
depth, and semantic ERP sensors.  Therefore this reproduction uses the native
path and intentionally does not insert cubemap interpolation into the main
experiment.  A clear error is raised if another environment lacks the API.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from omegaconf import OmegaConf, open_dict


RGB_UUID = "equirect_rgb_sensor"
SEMANTIC_UUID = "equirect_semantic_sensor"
DEPTH_UUID = "equirect_depth_sensor"
PERSPECTIVE_UUID = "perspective_rgb_sensor"


def native_equirectangular_supported() -> bool:
    try:
        import habitat_sim

        return hasattr(habitat_sim, "EquirectangularSensorSpec")
    except ImportError:
        return False


def configure_equirectangular_sensors(
    habitat_config: Any,
    height: int,
    width: int,
    sensor_height: float = 0.88,
    include_semantic: bool = True,
    include_depth: bool = True,
    include_perspective: bool = True,
) -> Any:
    """Mutate an unlocked Habitat config to use native full-sphere sensors.

    Call this inside Habitat's ``read_write(config)`` context.  Cloning an
    existing structured camera entry preserves all 0.3.3-required fields while
    changing only the registered sensor type, resolution and pose.
    """

    if not native_equirectangular_supported():
        raise RuntimeError(
            "This Habitat-Sim build lacks EquirectangularSensorSpec; "
            "the audited 0.3.3 native-sensor path is unavailable."
        )
    agent = habitat_config.simulator.agents.main_agent
    existing = agent.sim_sensors
    template = None
    for key in ("rgb_sensor", "semantic_sensor", "depth_sensor"):
        if key in existing:
            template = deepcopy(existing[key])
            break
    if template is None:
        raise RuntimeError("Habitat config has no camera sensor to use as a template")

    # EquirectangularSensorSpec does not expose pinhole-only ``hfov`` or
    # ``sensor_subtype``.  Construct the exact 0.3.3 SimulatorSensorConfig
    # fields rather than cloning those incompatible camera keys.
    rgb = OmegaConf.create(
        {
            "type": "HabitatSimEquirectangularRGBSensor",
            "uuid": RGB_UUID,
            "height": int(height),
            "width": int(width),
            "position": [0.0, float(sensor_height), 0.0],
            "orientation": [0.0, 0.0, 0.0],
        }
    )
    sensors = {RGB_UUID: rgb}
    if include_depth:
        depth = OmegaConf.create(
            {
                "type": "HabitatSimEquirectangularDepthSensor",
                "uuid": DEPTH_UUID,
                "height": int(height),
                "width": int(width),
                "position": [0.0, float(sensor_height), 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "min_depth": 0.0,
                "max_depth": 10.0,
                "normalize_depth": False,
            }
        )
        sensors[DEPTH_UUID] = depth
    if include_perspective:
        perspective = deepcopy(template)
        perspective.type = "HabitatSimRGBSensor"
        with open_dict(perspective):
            perspective.uuid = PERSPECTIVE_UUID
        perspective.height = int(height)
        perspective.width = int(width)
        perspective.position = [0.0, float(sensor_height), 0.0]
        perspective.orientation = [0.0, 0.0, 0.0]
        sensors[PERSPECTIVE_UUID] = perspective
    if include_semantic:
        semantic = OmegaConf.create(
            {
                "type": "HabitatSimEquirectangularSemanticSensor",
                "uuid": SEMANTIC_UUID,
                "height": int(height),
                "width": int(width),
                "position": [0.0, float(sensor_height), 0.0],
                "orientation": [0.0, 0.0, 0.0],
            }
        )
        sensors[SEMANTIC_UUID] = semantic
    agent.sim_sensors = sensors
    return habitat_config


def extract_panorama(observations: dict[str, Any]) -> Any:
    if RGB_UUID not in observations:
        raise KeyError(f"Missing native panorama observation {RGB_UUID!r}")
    return normalize_native_equirectangular(observations[RGB_UUID])


def extract_depth_panorama(observations: dict[str, Any]) -> np.ndarray:
    """Return horizontally normalized metric ERP depth as an ``HxW`` array."""

    if DEPTH_UUID not in observations:
        raise KeyError(f"Missing native panorama depth observation {DEPTH_UUID!r}")
    depth = np.asarray(normalize_native_equirectangular(observations[DEPTH_UUID]))
    depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW ERP depth, got {depth.shape}")
    return depth.astype(np.float32, copy=False)


def normalize_native_equirectangular(observation: Any) -> Any:
    """Convert Habitat-Sim native ERP ordering to the EAGOR convention.

    Audited Habitat-Sim 0.3.3 places perspective-image right at increasing ERP
    ``u``.  EAGOR uses positive azimuth/``+y`` for agent-left so that a positive
    yaw error maps directly to ``turn_left``.  A horizontal flip aligns RGB and
    semantic panoramas while leaving front at the center and back at the seam.
    This adapter was verified against a co-located perspective sensor on the
    bundled van-gogh-room scene.
    """

    if hasattr(observation, "flip") and not isinstance(observation, np.ndarray):
        # Torch tensors use dimension rather than axis.
        try:
            return observation.flip(dims=(-2,))
        except TypeError:
            pass
    return np.flip(np.asarray(observation), axis=1).copy()
