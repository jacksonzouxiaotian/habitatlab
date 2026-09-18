# Formal MP3D experiment environment

Snapshot date: 2026-09-05 (Asia/Shanghai).

## Software and hardware

| Component | Version |
|:---|:---|
| Repository revision before the local experiment changes | `d47cb51f2dbd77b1402a2a78f0eae12462935272` |
| Branch | `narrow-passage-rl-memory` |
| Python | 3.9.19 |
| Habitat-Lab | 0.3.3, editable source checkout |
| Habitat-Sim | 0.3.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| cuDNN | 9.10.2 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0 |
| Gym | 0.23.0 |
| Gymnasium | 1.1.1 |
| Stable-Baselines3 / sb3-contrib | 2.7.1 / 2.7.1 |
| GPU | NVIDIA RTX A4000, 16,376 MiB |
| NVIDIA driver | 570.211.01 |

The formal runs used `/home/xiaotian/miniconda3/envs/habitat/bin/python`.
`pytest` is not installed in that runtime, so the two new selector test files
were checked by direct import/smoke calls rather than reported as a completed
pytest suite.

## Dataset and checkpoint fingerprints

| Artifact | SHA-256 |
|:---|:---|
| Official MP3D PointNav v1 `val.json.gz` | `28cfac6b3729489b460c33daf93ed46f08c8975853df0f37b835b5b9a4f18d14` |
| MP3D-derived narrow-passage train (400 episodes) | `9ee0db9a1cda52e12c1b69dd24b21c31b50c8f598a115d286911b78af077177f` |
| MP3D-derived narrow-passage val (80 episodes) | `adc9dfe2d63542178e9eb27304da4dc21aabb7315e0c2198f21f7853c23eab8d` |
| Released MP3D RGB-D PointNav PPO checkpoint | `8b327d56d6fff4b3a711ff7768b56b4ffcf1d6796ea1ff877f97e16f1c84e363` |
| Released MP3D Depth backbone checkpoint | `54a2fedb9f430da9577c2b5f6aca34bb3a338fe2955a190f0d94f198a0f73f09` |
| Released Gibson-2+ Depth DD-PPO checkpoint | `a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad` |
| DEGNAV-E2E best checkpoint, seed 1701 | `086566050e223bdec74851e2c6a415731b505b19e56208b252f426d9d8aa02b8` |
| DEGNAV-E2E best checkpoint, seed 1702 | `d23fe975edb5b30208610af69f09e767054a0c44269710112835c1da3b3a73cd` |
| DEGNAV-E2E best checkpoint, seed 1703 | `ab42600af18b5a08a26541cfa763604f25a9d92cc024633c2c7dfc4e0f5c7cb0` |

The MP3D scene assets are licensed data and are not copied into the Git
repository. Reproduction requires their original `.glb` and `.navmesh` files
under a Habitat-compatible Matterport3D scene layout, plus the three released
baseline/initialization checkpoints identified above. The final three hashes
identify the locally generated DEGNAV-E2E best checkpoints.

## Runtime warnings

Habitat imports emit Gym's upstream maintenance warning, but these runs use
NumPy 1.26.4 rather than NumPy 2.x. Habitat-Sim also prints duplicate static
plugin notices during scene changes; no run failed because of those notices.
