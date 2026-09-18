# Failure-aware Memory for Geometry-Constrained Navigation

> 文档定位：AAAI / IROS Method 章节技术稿
> 当前实现：非参数、geometry-guided episode memory
> 证据边界：现有结果来自 synthetic recurrence / transfer protocol；不代表广泛真实环境泛化

## 1. Motivation

传统局部导航通常采用无记忆决策链：

```text
Current Observation -> Planner -> Action
```

该范式默认当前观测已经包含做出正确决策所需的全部信息。然而，窄通道导航具有明显的部分可观测性：入口处的 RGB-D 或局部点云可能显示足够的开口宽度，但内部转弯、遮挡、非对称障碍或机器人姿态约束可能使通道实际不可通过。若规划器只使用当前观测，机器人会把每次遭遇视为新问题，重复进入已经发生过碰撞、卡滞或超时的相似通道。

Failure-aware Memory 的目标不是简单保存完整轨迹，也不是提高 one-shot nominal passable-anchor success。它面向跨 episode 的失败复用问题：利用历史几何、动作和结果，估计当前 passage 的通过概率、碰撞风险和恢复可行性，从而减少重复不可行 commitment，同时避免对相似但可行的 passage 产生错误拒绝。

本文将导航决策扩展为：

```text
Current Geometry
      |
      v
Top-K Memory Retrieval
      |
      v
Confidence-aware Aggregation
      |
      v
Passability / Collision / Recovery Prediction
      |
      v
Commit / Explore / Recover / Reject
```

## 2. Memory Representation

### 2.1 Memory Item

记忆库记为 `M_t = {m_i}_{i=1}^{N_t}`。每个 memory item 保存一次 passage encounter：

```text
m_i = {
  geometry_feature,
  geometry_embedding,
  passage_width,
  robot_pose_relative_to_passage,
  action_sequence,
  failure_type,
  outcome,
  confidence,
  scene_id,
  timestamp
}
```

推荐字段定义如下：

| Field | Representation | Purpose |
|---|---|---|
| Geometry Feature | 19-D narrow-passage feature 或局部 3D geometry token | 描述 passage width、左右 clearance、heading、lateral offset 和 stuckness |
| Geometry Embedding | `z_i = f_g(g_i)` | 用于紧凑检索；当前规则实现也可直接使用离散 fingerprint |
| Passage Width | 米制宽度及 width/body ratio | 约束机器人形态相关的可行性 |
| Robot Pose | 相对入口的 `(x, y, yaw)` | 避免依赖不可迁移的全局坐标 |
| Action Sequence | `[(v_x, w_z)]_{1:T}` 或统计摘要 | 区分未尝试、振荡、强行进入和恢复行为 |
| Failure Type | collision / stuck / timeout / oscillation / wrong entry | 支持不同风险头和恢复策略 |
| Outcome | success / recovery / reject / execution failure | 监督通过性与恢复结果 |
| Confidence | `[0, 1]` | 表示传感器、标签和检索可信度 |

当前仓库的 19-D geometry observation 包含近/远 depth sectors、左右 clearance、passage width、body margin、heading error、lateral offset、goal distance、当前速度、stuck score、collision flag 和 previous action。对于跨场景记忆，建议保存相对 passage frame 的 pose，而不是世界坐标，以提高几何迁移能力。

### 2.2 Action Sequence Encoding

原始动作序列长度可变，不宜直接作为固定维度检索键。可采用两种表示：

1. **可解释统计量**：累计前进距离、最大角速度、角速度符号切换次数、倒退次数、低速度占比和最终姿态误差。
2. **时间编码器**：使用轻量 GRU/Transformer 将动作和本体状态序列编码为 `z_a`。

当前非参数实现主要使用 episode outcome、steps 和 passage fingerprint。时间编码器属于后续可训练扩展，不能在没有训练结果时描述为已验证模块。

### 2.3 Failure and Outcome Labels

失败标签必须由可观测事件定义，不能由 `success == false` 直接推导：

```text
collision = explicit simulator/contact event
stuck = low effective displacement over a fixed window
timeout = episode budget exhausted
oscillation = repeated angular/heading sign switching
wrong_entry = committed to benchmark-labeled false-feasible passage
correct_reject = explicit Reject on an infeasible passage
false_reject = explicit Reject on a feasible passage
```

Timeout、stuck 和 collision 都是执行失败，不应被重新标记为安全拒绝。

### 2.4 Confidence

Memory confidence 可由三部分组成：

```text
c_i = c_sensor * c_label * c_recency
```

- `c_sensor`：由 depth validity、dropout 和 geometry uncertainty 决定。
- `c_label`：真实 collision/recovery 标签高于仅由 timeout 推断的弱标签。
- `c_recency`：在环境变化或机器人形态更新后衰减旧记忆。

置信度应参与检索权重，但不能覆盖 passability 标签；低置信失败更适合触发 Explore，而不是直接 Reject。

## 3. Retrieval Mechanism

### 3.1 Query Construction

当前 observation 构成查询：

```text
q_t = {
  z_t,
  passage_width_t,
  width_body_ratio_t,
  entrance_angle_t,
  clearance_asymmetry_t,
  relative_pose_t
}
```

其中 `z_t = f_g(g_t)` 为当前 geometry embedding。若没有训练 encoder，可使用当前仓库中的 coarse fingerprint：

```text
fingerprint = (
  passage_width_bucket,
  approach_difficulty_bucket,
  corridor_class
)
```

### 3.2 Geometry-guided Similarity

建议的连续相似度为：

```text
sim(q_t, m_i) =
    w_z * cosine(z_t, z_i)
  - w_w * |width_ratio_t - width_ratio_i|
  - w_y * |entrance_angle_t - entrance_angle_i|
  - w_a * |clearance_asymmetry_t - clearance_asymmetry_i|
  - w_p * ||relative_pose_t - relative_pose_i||
```

该相似度显式区分“视觉或 embedding 相似”与“机器人形态条件下的几何可行性相似”。这也是 geometry-guided memory 与 vanilla episodic memory 的主要差异。

当前 `CrossEpisodeMemory` 使用离散 fingerprint 和 L-infinity radius 检索；`geometry_similarity` 提供连续几何相似度接口。二者均属于非参数检索，不应写成已训练 attention memory。

### 3.3 Top-K Retrieval and Aggregation

选择相似度最高的 `K` 个有效记忆：

```text
N_K(q_t) = TopK({sim(q_t, m_i)})
```

置信度加权系数为：

```text
alpha_i =
  exp(sim(q_t, m_i) / temperature) * c_i
  / sum_j exp(sim(q_t, m_j) / temperature) * c_j
```

由此得到：

```text
p_pass = sum_i alpha_i * I(outcome_i == success)

p_collision = sum_i alpha_i * I(failure_type_i == collision)

p_recovery = (
  sum_i alpha_i * I(outcome_i == recovery_success)
  / max(sum_i alpha_i * I(recovery_attempt_i), epsilon)
)
```

`p_recovery` 表示发生相似失败后执行恢复动作的条件成功概率，而不是无条件 passage success。

同时计算 memory uncertainty：

```text
u_memory =
  predictive_variance
  + 1 / max(effective_sample_size, 1)
```

当没有足够邻居时，系统应回退到 geometry prior，并提高 Explore 倾向，不能把空记忆解释为零风险。

## 4. Decision Module

### 4.1 Risk Prediction

Memory aggregation输出：

```text
Passability          = p_pass
Collision Probability = p_collision
Recovery Probability  = p_recovery
```

综合风险可写为：

```text
risk_t =
  lambda_c * p_collision
  + lambda_f * (1 - p_pass)
  + lambda_u * u_memory
  + lambda_s * stuck_score
```

当前仓库的正式证据支持规则风险与非参数 memory aggregation；神经 `risk_head` 只是接口，尚不能在论文中声称已经训练并优于规则风险。

### 4.2 Mode Selection

| Mode | Condition | Behavior |
|---|---|---|
| Commit | `p_pass` 高、`p_collision` 低、uncertainty 低 | 对齐后稳定穿越 |
| Explore | 证据不足但风险尚可 | 低速探测、更新 geometry 和 memory |
| Recover | stuck/collision 已发生且 `p_recovery` 高 | 后退、旋转和重新对齐 |
| Reject | `p_pass` 低且失败证据充分 | 停止进入并请求全局重规划 |

为避免模式抖动，模式切换应使用不同的进入/退出阈值和最小驻留时间：

```text
tau_enter != tau_exit
mode_t remains active for at least H_min steps
```

Reject 必须同时满足最低邻居数、最低 confidence 和低 passability，防止单次噪声失败导致永久黑名单。

### 4.3 Memory Update

只有真实执行或明确传感事件才能更新 outcome memory：

1. 执行成功：写入 success。
2. collision/stuck/timeout：写入对应失败类型。
3. recovery attempt：记录 recovery success/failure。
4. Reject：记录 decision，但不伪装成 traversal outcome。

为防止自我强化偏差，仅由 memory 触发的 Reject 不应再次作为失败 traversal 写入 memory。

## 5. Training Objective

### 5.1 Current Non-parametric Objective

当前仓库的 geometry-guided memory 不通过梯度训练。它通过记录 episode outcome、更新 `D_min` 后验并聚合相似案例成功率完成在线更新。因此，现阶段论文应描述为 non-parametric failure memory，而不是 learned memory network。

### 5.2 Optional Trainable Extension

若进一步训练 geometry encoder 和预测头，可采用：

```text
L =
  lambda_pass * BCE(p_pass, y_pass)
  + lambda_col * BCE(p_collision, y_collision)
  + lambda_rec * BCE(p_recovery, y_recovery)
  + lambda_rank * L_retrieval
  + lambda_cal * L_brier
  + lambda_int * L_interference
```

其中：

- `L_retrieval`：使相同几何机制与相同 outcome 的样本更接近。
- `L_brier`：约束概率校准，避免只优化分类准确率。
- `L_interference`：惩罚对 similar-but-feasible passage 的错误拒绝。

训练样本必须按 scene、passage identity 和时间划分，避免同一 passage 的重复轨迹同时出现在 train/test。查询时只能使用当前时刻之前的 memory，防止 outcome leakage。

## 6. Evaluation Protocol

### 6.1 Protocol Groups

| Group | Purpose |
|---|---|
| One-shot passable | 检查 nominal navigation，不用于证明 memory |
| Same-passage repeated false-feasible | 检查是否减少重复失败 |
| Similar-but-new false-feasible | 检查 memory transfer |
| Similar-but-feasible | 检查过度泛化与 interference |
| Geometry noise / pose shift | 检查检索鲁棒性 |
| Cross-scene / robot morphology | 检查跨环境与跨机身泛化，当前待完成 |

### 6.2 Baselines

1. No memory。
2. Local intra-episode memory。
3. Vanilla episodic memory。
4. kNN failure memory。
5. Geometry-guided cross-episode failure memory。

### 6.3 Metrics

| Metric | Interpretation |
|---|---|
| Passable Success Rate | memory 是否破坏正常通过 |
| Passable False Reject Rate | 对可行 passage 的干扰 |
| False-feasible Reject R1 / Final | 首轮与学习后的拒绝能力 |
| Transfer Reject Rate | 对相似新不可行 passage 的迁移 |
| Wasted Attempts / Steps | 未正确拒绝造成的重复执行代价 |
| Retrieval Precision / Recall | 检索的失败案例是否相关 |
| Brier Score / ECE | `p_pass`、`p_collision` 的校准 |
| Recovery Success | 相似失败后的恢复有效性 |
| Query Latency / Memory Size | 在线部署效率 |

### 6.4 Current Evidence

当前 paper preset 使用 5 seeds、8 rounds，以及每个 seed 40 个 same false-feasible、40 个 transfer false-feasible 和 40 个 similar feasible passages。现有结果为：

| Method | Passable SR | Transfer reject | Interference false reject |
|---|---:|---:|---:|
| No memory | 90.0±2.2% | 0.0±0.0% | 0.0±0.0% |
| Vanilla episodic memory | 0.0±0.0% | 100.0±0.0% | 100.0±0.0% |
| kNN failure memory | 0.0±0.0% | 100.0±0.0% | 100.0±0.0% |
| Geometry-guided memory | 90.0±2.2% | 93.0±8.6% | 0.0±0.0% |

该结果说明 geometry constraints 在当前 synthetic protocol 中有助于区分相似不可行与相似可行 passage。它不证明广泛真实环境泛化，也不证明 memory 提高 one-shot Habitat success。

### 6.5 Required Ablations

1. w/o geometry similarity。
2. w/o confidence weighting。
3. w/o action history。
4. w/o `D_min` calibration。
5. Different `K` and retrieval radius。
6. No recency decay。
7. Oracle failure labels vs weak timeout labels。

所有方法必须使用相同 passage groups、seed、round 和 evaluation budget，并按 seed 报告 mean ± std。成功、Reject、collision、timeout 和 stuck 必须分别记录。

## 7. Paper-facing Summary

推荐论文表述：

> We introduce a geometry-guided failure memory that retrieves passage-centric prior outcomes and aggregates them into passability, collision, and recovery estimates. Unlike generic episodic retrieval, the memory explicitly conditions similarity on robot-relative passage geometry, which is intended to suppress repeated infeasible commitments without rejecting geometrically similar feasible passages.

当前证据支持的结论：

> In the synthetic recurrence and transfer protocol, geometry-guided memory preserves passable success while reducing repeated false-feasible attempts and avoiding the interference observed with generic memory baselines.

当前证据不支持的结论：

- Memory 提高 one-shot nominal Habitat success。
- Memory 已在真实四足机器人上验证。
- 神经 risk head 已训练并优于非参数 aggregation。
- Synthetic transfer 等价于广泛 real-world generalization。
