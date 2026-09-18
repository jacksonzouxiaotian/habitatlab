# DEGNav strict feasibility：OBB / dual-margin 审计（2026-08-18）

本文是当前可复现实现与结果的唯一说明。旧结果
`paper_dynamic_20260818_170356` 未被覆盖；它使用圆形 simulator body，不能与本文
OBB 结果混为同一实验。

2026-08-20 新增的显式 `Gamma_{Delta psi}` / `oplus` 递归 belief、posterior
confidence、kappa sweep、failure-mode 与跨 morphology 审计见
[`belief_filter_ablation.md`](belief_filter_ablation.md)。该更新不覆盖本文列出的
2026-08-18 正式结果。

> 2026-08-20 structural repair update：修复 memory outcome semantics、max-range
> ray 语义、主动 Explore、stuck/Recover、入口 waypoint 和 swept-OBB 动作投影后，
> 最新 42-scenario smoke 的 feasible Success/False Reject/Collision/Timeout 为
> `50.00% / 0% / 0% / 50.00%`。L/S 仍均为 `0/3`，因此严格工程 gate 未通过，
> 没有运行 phase-4 calibration 或三 seed。完整结果与所有剩余低成功率原因见
> [`repaired_success_diagnosis.md`](../results/ablation_feasibility/repaired_diagnosis_20260820_164118/repaired_success_diagnosis.md)。

## 根因与修复范围

审计确认了四个相互独立的结构问题：

1. `procedural_env_v2.py` 的碰撞与 clearance 使用 `radius=0.18 m` 圆，决策器却用
   `0.36 x 0.60 m` 矩形，旧 `passable` 甚至仅等价于 `blocker is None`。
2. approach observation 将待进入通道报告为 `passage_width=10 m`，决策器在首次
   物理接触前看不到通道宽度。
3. 无 Gym 安装时 fallback `Box` 将数组 lower bound 错建为全 0，负角速度被裁掉，
   机器人不能向正确方向旋转。
4. 初版 structural runner 漏掉 `CrossEpisodeMemory.record_episode()`；这会让 full
   与 no-memory 实际等价。该版本结果已作废，最终 runner 有专门的构造测试。

当前链路为：

```text
RobotMorphology(width=.36, length=.60, safety_margin=.03)
  -> procedural_env_v2 OBB wall / circle-vs-OBB blocker collision
  -> OBB ground-truth passable + clearance + prospective entry observation
  -> DynamicFeasibilityEstimator
       structural mu/sigma/p/LCB/UCB
       pose-conditioned mu/sigma/p/LCB/UCB
  -> select_ablation_mode
  -> TurnCommitFSM / fsm_action
       Reject / Align-or-Explore / Commit / Recover
  -> paired linear/angular action + per-step audit log
  -> episode outcome -> CrossEpisodeMemory.record_episode (except no_memory)
```

环境、ground-truth label、clearance 和 strict controller 持有同一个不可变
`RobotMorphology` 对象。`robot_radius` 仅保留为只读绘图兼容属性，不再参与 strict
碰撞或标签。

## 几何与决策公式

结构 margin：

```text
W_struct = body_width + 2*safety_margin
delta_struct = passage_width - W_struct
```

矩形在通道法向上的真实投影必须使用加号：

```text
W_pose(yaw) = body_width*abs(cos(yaw))
            + body_length*abs(sin(yaw))
            + 2*safety_margin
delta_pose = passage_width - W_pose(yaw)
```

请求中出现过的减法式会在 45 度附近产生负 required width，且 safety margin 越大
required width 反而越小，因此与 OBB 几何矛盾，未采用。

动态 uncertainty 分别输出 structural 与 pose 路径，输入包括 depth ray dispersion、
valid/dropout ratio、boundary-fit residual、temporal width variation、yaw/lateral pose
uncertainty。两条路径均使用：

```text
LCB = mu_delta - kappa*sigma_delta
UCB = mu_delta + kappa*sigma_delta
```

共享阈值保持原声明值，未针对结果调整：`kappa=1.645`，
`tau_commit=tau_reject=0.02 m`。

- Reject：仅 `UCB_struct < -tau_reject`、明确 blocker 或重复失败 memory。
- Align/Explore：structural 尚不确定，或 structural 可行但 pose 尚未就绪；入口 CI
  overlap 时不以正线速度盲目前探。
- Commit：`LCB_struct > tau_commit` 且 `LCB_pose > tau_commit`。
- Recover：collision、stuck 或 prior commitment failure。

## 主消融定义

| 方法 | uncertainty 对决策 | yaw-aware readiness | failure memory |
|:---|:---|:---:|:---:|
| `full_dynamic_uncertainty` | dynamic structural/pose interval gate | 是 | 是 |
| `mean_only` | estimator 保留并记录；selector 只读两条 mean | 是 | 是 |
| `fixed_uncertainty` | 两条路径使用全局 `sigma=0.05 m` | 是 | 是 |
| `no_yaw_aware_readiness` | dynamic interval gate | 否；其余不变 | 是 |
| `no_memory` | 与 full 相同 | 是 | 否 |

`point_estimate` 和 `no_uncertainty` 都是 `mean_only` 的兼容别名。分析脚本会将它们
折叠到 `mean_only`，并在 `compatibility_alias_audit.csv` 标记“行为等价、不得作为
两个独立论文结果”。旧 `full`、`no_yaw_prior` 也仅是输入兼容别名。

## 每步审计字段

`steps.csv` 含 observation/random hash、linear/angular action，以及：

```text
mu_delta_struct, sigma_delta_struct, p_feas_struct,
LCB_struct, UCB_struct,
mu_delta_pose, sigma_delta_pose, p_feas_pose,
LCB_pose, UCB_pose,
yaw_error, W_required_with_yaw, W_required_without_yaw,
ray_dispersion, valid_depth_ratio, dropout_ratio,
boundary_fitting_residual, temporal_width_variation, yaw_pose_sigma,
uncertainty_triggered, yaw_prior_changed_decision,
explicit_blocker, feasibility_mode, selected_mode,
linear_action, angular_action
```

## 单元与 mechanism validation

最终整个 `examples/narrow_passage_rl/tests` 套件：

```text
48 passed in 0.17s
```

其中 strict 定向子集为 28 tests，并包含 yaw=`0/30/45/60` 的 OBB 墙面边界、统一 passable label、共享 morphology
对象、fallback action bounds、large-yaw `Align -> Commit`、truly-narrow Reject、
pose infeasible 不得直接 Reject、scene-dependent sigma、full-vs-mean 高不确定性分歧、
别名信息屏障以及 no-memory 真正切断 memory 路径。

45 个 paired mechanism episodes 的结果：

| 指标 | 结果 |
|:---|---:|
| dynamic full sigma distinct values | 132 |
| full uncertainty trigger | 34/180 = 18.89% |
| full-vs-mean feasibility disagreement | 34/180 = 18.89% |
| full-vs-mean final mode disagreement | 22/180 = 12.22% |
| yaw readiness changed decision | 22/180 = 12.22% |
| observation/random hashes | 全部一致 |

有效目录：

```text
results/ablation_feasibility/mechanism_20260818_195905/
```

## 最终 closed-loop smoke

最终代码上的 42 个 paired scenarios smoke 全部 gate 通过：geometry/unit tests、
large-yaw Align→Commit、truly-narrow Reject、非退化 Success、bin 覆盖和 paired hash。

| 方法 | Feasible Success | Feasible Collision | Feasible False Reject | Infeasible Correct Reject |
|:---|---:|---:|---:|---:|
| full dynamic | 11.1% | 16.7% | 61.1% | 45.8% |
| mean only | 5.6% | 22.2% | 72.2% | 75.0% |
| fixed uncertainty | 16.7% | 16.7% | 61.1% | 66.7% |
| no yaw-aware readiness | 11.1% | 16.7% | 61.1% | 45.8% |
| no memory | 11.1% | 22.2% | 0.0% | 16.7% |

有效目录：

```text
results/ablation_feasibility/structural_smoke_20260818_195309/
```

## 三 seed 正式评测

协议按六个 structural margin bins、七个 scene、三个 seed 分层。每个
`bin x scene x seed` 有 5 个 paired scenarios 并覆盖全部 yaw；noise 和 lateral
使用平衡设计：

```text
margin bins = [-.15,-.10),[-.10,-.05),[-.05,0),[0,.05),[.05,.10),[.10,.15]
yaw          = 0,15,30,45,60 deg       (各 126 scenarios)
noise        = clean,mild,severe        (各 210 scenarios)
lateral      = 0,.10,.20 m              (各 210 scenarios)
scenarios    = 630 paired
episode rows = 3150 = 630 x 5 methods
step rows    = 129414
```

负 structural margin 按定义不可能是 OBB-passable；正 bin 中六个普通 scene 可行，
`false_feasible` 提供 blocker 导致的不可行样本。因此要求“负 margin bin 内也有
passable”会与 ground-truth 定义矛盾。

总体指标（分母包含 feasible 与 infeasible）：

| 方法 | Success | Collision | Correct Reject | False Reject | Timeout/Stuck |
|:---|---:|---:|---:|---:|---:|
| full dynamic | 0.2% | 6.3% | 46.7% | 41.4% | 5.4% |
| mean only | 1.1% | 9.0% | 49.2% | 40.5% | 0.2% |
| fixed uncertainty | 0.3% | 7.8% | 48.6% | 41.3% | 2.1% |
| no yaw-aware readiness | 0.2% | 7.0% | 46.3% | 41.4% | 5.1% |
| no memory | 5.2% | 18.6% | 15.1% | 0.0% | 61.1% |

按 label 分组后才是主要解释：

| 方法 | Feasible Success | Feasible Collision | Feasible False Reject | Infeasible Correct Reject | Infeasible Wasted Commitment |
|:---|---:|---:|---:|---:|---:|
| full dynamic | 0.4% | 1.5% | 96.7% | 81.7% | 12.2% |
| mean only | 2.6% | 3.0% | 94.4% | 86.1% | 17.8% |
| fixed uncertainty | 0.7% | 1.9% | 96.3% | 85.0% | 15.6% |
| no yaw-aware readiness | 0.4% | 1.9% | 96.7% | 81.1% | 13.9% |
| no memory | 12.2% | 21.5% | 0.0% | 26.4% | 17.8% |

### 可汇报结论

- 动态 uncertainty 和双 interval gate 确实进入决策路径；这已由非零 mechanism
  disagreement 证明。
- 当前 full **不优于** mean/fixed；Success 的 paired CI 不支持优势声明。
- yaw-aware readiness 对正式 Success 的差为 0.00 percentage points；不能声称其
  提升了 Success，只能报告它改变 required width/模式并小幅改变碰撞。
- 最强信号来自 memory：相对 no-memory，full 的总体 False Reject 增加
  41.43 percentage points（95% CI 39.38–43.48），Success 降低 5.08 points
  （95% CI 3.27–6.89）。failure memory 将 timeout/collision 泛化成后续结构 Reject，
  造成 feasible subset 96.7% False Reject。
- no-memory 不是“更安全的赢家”：它降低 False Reject 并提高 Success，但带来
  更高 collision 和 66.3% feasible timeout。结论是 memory 语义需要单独重设计，
  不是调 feasibility 阈值。

最终可汇报目录：

```text
results/ablation_feasibility/structural_full_20260818_194833_final/
  episodes.csv
  steps.csv
  gate_report.json
  run_metadata.json
  subset_metrics.csv
  analysis/overall_ablation_table.md
  analysis/label_conditional_metrics.csv
  analysis/margin_phase.{csv,png,pdf}
  analysis/yaw_sensitivity.{csv,png,pdf}
  analysis/risk_coverage.{csv,png,pdf}
```

`paper_dynamic_20260818_170356` 仍可作为“旧圆形几何为何产生 0.2%–0.3%
Success”的历史诊断，但不是本文 OBB 正式结果。`structural_full_20260818_192725`
是中止的单进程 partial；`structural_full_20260818_192825` 与其派生的
`194248_validated` 漏 memory update，不得汇报。

## 低 Success 的完整诊断（2026-08-20）

最新诊断不再把低值只概括为 memory overgeneralization，而是区分三层原因：

1. 630 个 mixed episodes 中 360 个 ground-truth infeasible，不能进入 Success 分子；
2. full 在 270 个 feasible episodes 中由外层 memory false-reject 261 个；
3. 去掉 memory 后，open-space max-range/dropout uncertainty、零速度 Explore、stuck
   与反向 Recover 形成闭环，产生 66.30% Timeout；scene-specific controller 仍产生
   21.48% Collision。

逐项因果证据、memory 排序/写回问题、scene 表、noise/yaw/margin 分解与已排除嫌疑见：

```text
results/ablation_feasibility/success_rate_diagnosis_20260820_150459/
  success_rate_diagnosis.md
  cause_registry.csv
  method_outcomes.csv
  memory_order_audit.csv
  open_space_uncertainty.csv
  timeout_cycle.csv
  no_memory_scene_outcomes.csv
  no_memory_factor_outcomes.csv
  mode_mechanisms.csv
```

## 完整复现命令

所有输出必须使用新时间戳；脚本拒绝覆盖已有文件。

```bash
PY=/home/xiaotian/miniconda3/envs/navila/bin/python
export PYTHONPATH=examples/narrow_passage_rl

# 1. Unit tests
$PY -m pytest -q \
  examples/narrow_passage_rl/tests/test_feasibility_ablation.py \
  examples/narrow_passage_rl/tests/test_strict_obb_geometry.py

# 2. 45-episode mechanism validation
$PY examples/narrow_passage_rl/eval_feasibility_stress.py \
  --episodes 45 \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/mechanism_NEW_TIMESTAMP

# 3. Closed-loop smoke
$PY examples/narrow_passage_rl/eval_structural_feasibility.py \
  --phase smoke --seeds 0 --max-steps 200 \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/structural_smoke_NEW_TIMESTAMP

# 4. Combine all gates
$PY examples/narrow_passage_rl/validate_structural_feasibility.py \
  --mechanism-report .../mechanism_NEW_TIMESTAMP/mechanism_report.json \
  --smoke-report .../structural_smoke_NEW_TIMESTAMP/gate_report.json \
  --output .../structural_smoke_NEW_TIMESTAMP/validation_report.json

# 5. Only when validation_report says full_evaluation_permitted=true.
# The simplest reproducible path runs all seeds in one process:
$PY examples/narrow_passage_rl/eval_structural_feasibility.py \
  --phase full --seeds 0 1 2 --max-steps 200 --factor-design balanced \
  --gate-report .../validation_report.json \
  --output-dir .../structural_full_NEW_TIMESTAMP

# 6. Analysis
$PY examples/narrow_passage_rl/analyze_feasibility_ablation.py \
  --input-csv .../structural_full_NEW_TIMESTAMP/episodes.csv \
  --output-dir .../structural_full_NEW_TIMESTAMP/analysis
```

For parallel execution, run three otherwise identical commands with one seed,
add `--seed-shard`, write to `.../seed0`, `.../seed1`, `.../seed2`, then run:

```bash
$PY examples/narrow_passage_rl/merge_structural_feasibility.py \
  --root .../structural_full_SHARDS_TIMESTAMP \
  --output-root .../structural_full_NEW_TIMESTAMP_final
```
