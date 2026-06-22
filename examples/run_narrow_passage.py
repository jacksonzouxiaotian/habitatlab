#!/usr/bin/env python3
"""Sanity-check a mined narrow-passage dataset against the real
NarrowPassageNav-v0 task before spending GPU time on training.

Loads habitat.dataset.data_path/split through the existing pointnav_habitat_test
benchmark config with the task swapped to narrow_passage (the same composition
ppo_narrow_passage.yaml uses), runs random actions for a handful of episodes,
and prints the narrow-passage measures plus live clearance per episode so
obviously broken episodes (NaNs, constant zero success, no movement) are
caught early.

Run inside the `habitat` conda env. Example:
    python examples/run_narrow_passage.py \
        --data-path data/datasets/narrow_passage/{split}/{split}.json.gz \
        --split val --num-episodes 10
"""

import argparse

import habitat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="data/datasets/narrow_passage/{split}/{split}.json.gz",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()

    # Resolve {split} ourselves: Hydra's override-string grammar treats
    # literal "{" as a dict-literal token, so passing the placeholder through
    # to compose() raises an OverrideParseException.
    data_path = args.data_path.format(split=args.split)
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={args.split}",
            "habitat.dataset.type=PointNav-v1",
        ],
    )

    with habitat.Env(config=config) as env:
        # env.action_space.sample() is unusable here: VelocityAction.action_space
        # (habitat-lab/habitat/tasks/nav/nav.py) wraps its two Box args in
        # habitat's custom ActionSpace, whose .sample() picks only ONE of
        # {linear_velocity, angular_velocity} rather than both. Sample the
        # two Box spaces directly instead.
        vel_spaces = env.action_space.spaces["velocity_control"].spaces

        num_episodes = min(args.num_episodes, env.number_of_episodes or args.num_episodes)
        for ep_idx in range(num_episodes):
            env.reset()
            episode = env.current_episode
            steps = 0
            while not env.episode_over and steps < args.max_steps:
                action = {
                    "action": "velocity_control",
                    "action_args": {
                        "linear_velocity": float(vel_spaces["linear_velocity"].sample()[0]),
                        "angular_velocity": float(vel_spaces["angular_velocity"].sample()[0]),
                    },
                }
                env.step(action)
                steps += 1

            metrics = env.get_metrics()
            np_state = env.task.get_narrow_passage_state()
            print(
                f"episode {episode.episode_id} (scene={episode.scene_id}) "
                f"steps={steps} "
                f"success={metrics.get('narrow_passage_success')} "
                f"collision={metrics.get('narrow_passage_collision')} "
                f"stuck={metrics.get('narrow_passage_stuck')} "
                f"final_heading_error={np_state.heading_error:.3f} "
                f"final_lateral_offset={np_state.lateral_offset:.3f} "
                f"final_stuck_score={np_state.stuck_score:.3f}"
            )


if __name__ == "__main__":
    main()
