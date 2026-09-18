# Open-Nav Install Status

This document records the local Open-Nav installation status for using
Open-Nav as a future zero-shot VLN baseline.

## Local Paths

```text
Open-Nav source:
/home/xiaotian/vla/Open-Nav

Shared VLN data root:
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln

Preflight report:
examples/narrow_passage_rl/results/opennav_preflight/opennav_preflight.md
```

## Completed

| Item | Status |
|---|---:|
| Open-Nav repository clone | done, commit `3a8dcef` |
| Habitat 0.1.7 Python stack through `navila-eval` | pass |
| OpenNav_R2R-CE_100 quick dataset | present, 100 episodes |
| OpenNav val-unseen GT | present |
| DDPPO depth encoder | present via shared VLN data |
| RAM source repository | cloned and editable-installed |
| Python lightweight dependencies | installed in `navila-eval` |
| Open-Nav preflight script | implemented |

## Current Blockers

| Item | Status / note |
|---|---|
| MP3D scenes | missing 10/10 scenes for OpenNav_R2R-CE_100 quick set |
| waypoint predictor checkpoint | Google Drive download started but remained incomplete |
| RAM checkpoint | public HF file found, but large-file download was too slow in this session |
| SpatialBot3B | HuggingFace repo is gated; requires authenticated access |
| LLM backend | `OPENAI_API_KEY` is not set, or a local OpenAI-compatible endpoint must be configured |

The missing quick-set MP3D scenes are:

```text
2azQ1b91cZZ
8194nk5LbLH
EU6Fwq7SyZv
QUCTc6BB5sX
TbHJrupSAjP
X7HyMhZNoso
Z6MFQCViBuw
oLBMNvg9in8
x8F5xyUWy9e
zsNo4HB9uLZ
```

## Preflight Command

```bash
PYTHONPATH=/home/xiaotian/vla/Open-Nav:/home/xiaotian/vla/habitat-lab-v0.1.7 \
conda run -n navila-eval \
python examples/narrow_passage_rl/opennav_preflight.py
```

## Run Command After All Blockers Are Resolved

```bash
cd /home/xiaotian/vla/Open-Nav
PYTHONPATH=/home/xiaotian/vla/Open-Nav:/home/xiaotian/vla/habitat-lab-v0.1.7 \
conda run -n navila-eval bash run_OpenNav.bash
```

## Paper Use

Open-Nav should be treated as a planned zero-shot open-source LLM VLN baseline
until the preflight reports `PASS` and paired simulator rollouts produce
standard metrics.  Do not report SR, SPL, NE, nDTW, or SDTW from the current
install-only state.
