# DEGNav strict feasibility：显式递归 belief 与泛化审计（2026-08-20）

本文承接 `feasibility_ablation.md` 的 OBB / dual-margin 定义。已有正式结果
`structural_full_20260818_194833_final` 和历史圆形结果
`paper_dynamic_20260818_170356` 均未覆盖；所有新增实验写入独立时间戳目录。

## 调用链与状态边界

完整在线链路为：

```text
observation[19]
  -> DynamicFeasibilityEstimator.update
       width EWMA + engineering sigma
       structural / pose recursive posterior (audit-only)
  -> TurnCommitFSM._strict_step_inputs
       legacy mu_delta_* / sigma_delta_* -> LCB/UCB
  -> select_ablation_mode
  -> fsm_action / TurnCommitFSM
  -> Commit / Explore(Align) / Recover / Reject
  -> linear/angular action + steps.csv
```

审计发现当前代码中不存在所谓“`DynamicFeasibilityEstimator` 的 outcome-based
update”。episode outcome 实际进入 `CrossEpisodeMemory.record_episode()`，后者再更新
`DMinCalibrator` 的离散 posterior 和 passage retrieval record。这是跨 episode failure
memory，不是逐步 structural/pose belief。本文没有把两套状态拼接或改名，以免产生
无法审计的因果泄漏。

## 改动前

旧均值是 passage width 的 EWMA：

```text
mu_D,t = (1-alpha) * mu_D,t-1 + alpha * D_t
mu_struct,t = mu_D,t - W_struct
mu_pose,t = mu_D,t - W_pose(yaw_t)
alpha = 0.55
```

旧 `sigma_delta_*` 是 ray dispersion、dropout、boundary residual、temporal width
variation 与 yaw/lateral pose uncertainty 的工程组合；它不是递归 posterior variance。

## 改动后：显式两步滤波

定义 margin belief：

```text
q_t(delta) = Normal(mu_t, P_t)
delta = D - W(yaw)
```

第一步，yaw 变化下的传播算子：

```text
q_t^- = Gamma_{Delta psi}[q_t-1]
mu_t^- = mu_t-1 - (W_t - W_t-1)
P_t^- = P_t-1 + Q_t
```

structural 路径的 `W_t-W_t-1=0`；pose 路径使用 OBB 精确投影：

```text
W_pose(yaw) = width*abs(cos(yaw))
            + length*abs(sin(yaw))
            + 2*safety_margin

Q_struct = sigma_process^2
Q_pose = Q_struct + (g_yaw * abs(dW_pose/dyaw) * wrap(Delta yaw))^2
```

第二步，观测融合算子：

```text
q_t = q_t^- oplus r_t
mu_t = (1-K_t)*mu_t^- + K_t*z_t
P_t = (1-K_t)^2*P_t^- + K_t^2*R_t
K_t = alpha = 0.55
```

其中 `z_t=D_t-W_t`，`R_t` 使用对应路径的 dynamic engineering variance。固定
`K_t=alpha` 使默认 posterior mean 与改动前 EWMA 代数等价；posterior variance 是
新增的一阶递归分布统计量。

## 因果隔离开关与兼容性

| 开关 | 默认 | 关闭时行为 | selector 是否读取 |
|:---|:---:|:---|:---:|
| `enable_belief_propagation` | on | `Gamma` 返回上一 posterior，不施加 yaw required-width shift / process variance | 否 |
| `enable_belief_fusion` | on | `oplus` 返回 propagated prior，不融合当前观测 | 否 |

两个开关既可写入 `DynamicFeasibilityConfig`，也可作为 `update()` 的可选 keyword
逐步覆盖；stress/structural runner 同时暴露
`--disable-belief-propagation` 和 `--disable-belief-fusion`。原调用方不传参数时 API
与行为不变。selector 继续只读旧
`mu_delta_* / sigma_delta_*`；这使 propagate-only、fuse-only 后续消融不会在未声明
的情况下改变论文方法。

## 新增逐步字段

`steps.csv` 在旧工程字段之外新增：

```text
posterior_mu_delta_struct, posterior_var_delta_struct,
posterior_sigma_delta_struct, posterior_p_feas_struct,
posterior_concentration_struct,
posterior_mu_delta_pose, posterior_var_delta_pose,
posterior_sigma_delta_pose, posterior_p_feas_pose,
posterior_concentration_pose,
propagated_mu_delta_struct, propagated_var_delta_struct,
propagated_mu_delta_pose, propagated_var_delta_pose,
belief_mean_equivalence_error_struct, belief_mean_equivalence_error_pose,
belief_propagation_enabled, belief_fusion_enabled,
belief_initialized_from_observation, belief_update_count
```

分布集中度定义为：

```text
concentration = 1 / (1 + posterior_sigma / fixed_sigma_reference)
fixed_sigma_reference = 0.05 m
posterior_p_feas = Phi(posterior_mu / posterior_sigma)
```

它只由 posterior variance 推导，不覆盖旧 `sigma_delta`。

## 等价性与单元测试

隔离 ROS 第三方 pytest 插件后，用户指定的两个回归文件结果：

```text
42 passed in 0.28s
```

包含 balanced-protocol 等其他 narrow-passage tests 的完整 suite 为：

```text
64 passed in 0.25s
```

新增测试验证：

- `Gamma` 与 `oplus` 数值符合文档公式；
- 默认 structural/pose posterior mean 与旧 EWMA 每步误差小于 `1e-12`；
- propagate 与 fuse 可独立关闭，且不会改变 selector-facing 字段；
- posterior variance 不是常数，并与工程 sigma 并行存在；
- 0.28×0.55、0.45×0.65、0.36×0.80 三种体型在 yaw=0/30/45/60° 的
  OBB 墙面边界全部通过。

此外，把改动前 `mechanism_20260818_195905` 与改动后 900 个 keyed step rows
逐字段比较，旧 structural/pose mu、engineering sigma、LCB/UCB、feasibility mode、
final selected mode 和 linear/angular action 的 mismatch 全部为 0，最大数值差为
`0.0`。明细见新目录的 `equivalence_report.md`。

## 45-episode mechanism validation

有效目录：

```text
results/ablation_feasibility/mechanism_belief_20260820_141927/
```

| 指标 | 结果 |
|:---|---:|
| executed paired episodes | 45 |
| full steps | 180 |
| engineering sigma distinct values | 132 |
| uncertainty branch | 34/180 = 18.89% |
| full-vs-mean final mode disagreement | 22/180 = 12.22% |
| yaw prior changed decision | 22/180 = 12.22% |
| paired observation/random hashes | 全部一致 |

置信度相关性只分析 full 的逐步记录：

| confidence | selected-mode eta-squared | uncertainty-triggered r | r-squared | mean score |
|:---|---:|---:|---:|---:|
| engineering feature sigma | 0.050 | 0.381 | 0.145 | 0.098 |
| recursive posterior sigma | 0.047 | 0.350 | 0.123 | 0.085 |

一句话结论：本次 mechanism validation 中旧工程 sigma 的联合解释力略强于新增
posterior sigma（0.098 vs 0.085）；这是相关性诊断，不是因果效应。

## kappa / tau smoke 权衡

`tau_commit=tau_reject=0.02 m` 全程固定，只扫描 kappa。每个点使用同一套 42 个
paired smoke scenarios；下表三个比例均以其中 18 个 ground-truth feasible episodes
为分母。

有效目录：

```text
results/ablation_feasibility/kappa_sweep_20260820_140254/
```

| kappa | False Reject | Collision | Success |
|---:|---:|---:|---:|
| 0.500 | 66.67% | 16.67% | 16.67% |
| 1.000 | 61.11% | 16.67% | 16.67% |
| 1.280 | 61.11% | 11.11% | 16.67% |
| **1.645 (paper)** | **61.11%** | **16.67%** | **11.11%** |
| 2.000 | 61.11% | 16.67% | 11.11% |
| 2.500 | 61.11% | 16.67% | 5.56% |

这是小样本阶梯曲线，不支持把某个 kappa 宣称为最优；尤其论文点 1.645 没有在
三个指标上同时占优。`sweep.csv`、`kappa_tradeoff.png` 与 PDF 同目录保存。

## 既有正式结果的 failure-mode 量化

输入仍是只读的 `structural_full_20260818_194833_final`，输出写入：

```text
structural_full_20260818_194833_final/failure_modes_20260820_141220/
```

| 互斥类别 | Episodes | 所有 non-success 占比 | 类内 Success |
|:---|---:|---:|---:|
| correct infeasible reject（策略正确但导航 Success=False） | 294 | 46.74% | 0% |
| cross-episode memory-driven false reject | 261 | 41.49% | 0% |
| genuine OBB collision | 40 | 6.36% | 0% |
| yaw-readiness/alignment stall | 17 | 2.70% | 0% |
| conservative interval indecision | 12 | 1.91% | 0% |
| controller timeout (other) | 4 | 0.64% | 0% |
| depth-noise-driven boundary misjudgment | 1 | 0.16% | 0% |
| body-margin conversion offset | 0 | 0% | n/a |
| conservative-threshold false reject | 0 | 0% | n/a |
| explicit-blocker false positive | 0 | 0% | n/a |

类内 Success 为 0 是因为表按 `success=False` 条件筛选。总体最多的是策略正确的
infeasible reject；排除这类非错误样本后，主失效是 memory-driven false reject。
其直接审计判据是 `controller_mode=REJECT`、同时
`feasibility_mode!=REJECT`。日志没有独立 ground-truth yaw，因此没有把“大 yaw”
不诚实地命名为 yaw estimation error，而使用可观测的 alignment-stall 类别。

## 跨 morphology 三 seed 泛化

新增三种 morphology；每种先单独通过 42-scenario smoke gate，再运行 630 个
paired scenarios（3150 method episode rows）。原 0.36×0.60 结果只读复用。

有效目录：

```text
results/ablation_feasibility/morphology_generalization_20260820_140820/
```

| Morphology | full Success | mean Success | fixed Success | full False Reject | full-no_memory FR delta | full-no_yaw Success delta |
|:---|---:|---:|---:|---:|---:|---:|
| base 0.36×0.60 | 0.37% | 2.59% | 0.74% | 96.67% | +96.67 pp [94.44,98.52] | 0.00 pp |
| narrow 0.28×0.55 | 0.37% | 4.44% | 0.74% | 96.67% | +96.67 pp [94.44,98.52] | 0.00 pp |
| wide 0.45×0.65 | 0.00% | 1.48% | 0.00% | 97.41% | +97.41 pp [95.19,99.26] | 0.00 pp |
| long 0.36×0.80 | 0.37% | 3.33% | 0.74% | 96.30% | +96.30 pp [94.07,98.52] | 0.00 pp |

表中 Success / False Reject 均以每种 morphology 的 270 个 feasible episodes 为
分母。三个新运行各自具有 yaw=`126×5`、noise=`210×3`、lateral magnitude=`210×3`
的平衡计数，paired observation/random hash 全部一致。

结论：(a) memory 导致 False Reject 显著上升在四种体型上都复现；(b) yaw-aware
readiness 对 Success 的 paired 差值在四种体型上都精确为 0.00 pp。与此同时 full
Success 始终不超过 0.37%，所以这些是失败机制的泛化证据，不是 full 方法性能优势。

## 为什么 Success 仍然很低

完整逐步因果诊断已另存至：

```text
results/ablation_feasibility/success_rate_diagnosis_20260820_150459/
```

它确认低值由 mixed infeasible denominator、memory 的 outcome 语义/排序/自增强
false reject、open-space max-range 被当作 dropout、零信息 Explore、stuck→反向
Recover 循环以及 scene-specific controller failure 共同导致。新增 posterior belief
与旧决策逐步完全等价，因此不是本次低 Success 的来源。

## 复现命令

```bash
PY=/home/xiaotian/miniconda3/envs/navila/bin/python
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

$PY -m pytest -q \
  examples/narrow_passage_rl/tests/test_feasibility_ablation.py \
  examples/narrow_passage_rl/tests/test_strict_obb_geometry.py

$PY examples/narrow_passage_rl/eval_feasibility_stress.py \
  --episodes 45 \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/mechanism_belief_NEW_TIMESTAMP

$PY examples/narrow_passage_rl/analyze_belief_confidence.py \
  --input-csv .../mechanism_belief_NEW_TIMESTAMP/steps.csv \
  --output-dir .../mechanism_belief_NEW_TIMESTAMP

$PY examples/narrow_passage_rl/sweep_kappa_tau.py \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/kappa_sweep_NEW_TIMESTAMP

$PY examples/narrow_passage_rl/classify_feasibility_failures.py \
  --episodes-csv examples/narrow_passage_rl/results/ablation_feasibility/structural_full_20260818_194833_final/episodes.csv \
  --steps-csv examples/narrow_passage_rl/results/ablation_feasibility/structural_full_20260818_194833_final/steps.csv \
  --output-dir .../failure_modes_NEW_TIMESTAMP

$PY examples/narrow_passage_rl/eval_morphology_generalization.py \
  --execute --workers 3 \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/morphology_generalization_NEW_TIMESTAMP
```
