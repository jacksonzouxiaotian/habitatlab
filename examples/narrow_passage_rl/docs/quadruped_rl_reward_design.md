# Reward Design for Quadruped Navigation and Locomotion

> 文档定位：ICRA / IROS Method 章节技术稿
> 适用层级：Isaac Sim 四足 locomotion 与窄通道局部运动技能
> 说明：本文给出设计原则，不虚构尚未完成的 Isaac Sim 或 Lite3 实验结果

## 1. Reward Formulation

用户给出的形式：

```text
r =
  r_tracking
  + r_alive
  + r_forward
  + r_center
  - r_collision
  - r_energy
  - r_smooth
  - r_termination
```

该符号约定是合理的：tracking、alive、forward 和 center 是正向奖励，collision、energy、smooth 和 termination 是非负代价。实现时建议显式加入权重，并统一为“正奖励 + 显式代价”：

```text
r_t =
    w_tracking * r_tracking
  + w_alive    * r_alive
  + w_forward  * r_forward
  + w_center   * r_center
  - w_collision  * c_collision
  - w_energy     * c_energy
  - w_smooth     * c_smooth
  - w_termination * c_termination
```

所有 `r_*` 越大越好，所有 `c_*` 越大越差。该约定应在代码、配置、TensorBoard 和论文中保持一致。

## 2. Reward Components

### 2.1 Tracking Reward

Tracking reward 衡量机器人对高层速度、yaw 或姿态命令的跟踪：

```text
r_tracking =
  exp(-k_v * ||v_body - v_cmd||^2)
  * exp(-k_w * |w_z - w_cmd|^2)
```

作用：

- 建立 command-conditioned locomotion。
- 使 Commit、Explore 和 Recover 能共享一个低层策略。
- 保证导航层输出的速度命令具有可执行意义。

风险：

- 仅追踪速度可能让机器人在障碍前继续前进。
- 过大的 yaw tracking 权重可能诱发快速转体和足端打滑。

因此 tracking 必须与 collision、contact、clearance 和姿态约束共同使用。

### 2.2 Alive Reward

Alive reward 鼓励机器人保持非跌倒、非终止和姿态稳定：

```text
r_alive = I(robot_is_upright and episode_active)
```

作用：

- 为稀疏成功之前的长时运动提供稳定学习信号。
- 减少训练早期频繁跌倒。

潜在问题：

- 权重过大时，最优策略可能是原地站立直到超时。
- 若与 timeout penalty 不匹配，会形成“生存但不完成任务”的策略。

建议将 alive reward 保持较小，并同时使用 potential-based progress 和明确的超时成本。

### 2.3 Forward Progress Reward

Forward reward 应定义为到局部目标的势函数差，而不是单纯奖励机身前向速度：

```text
r_forward = d_goal(s_t) - d_goal(s_{t+1})
```

作用：

- 奖励真实任务进展。
- 允许 Recover 阶段短暂倒退，只要后续提高总体成功率。
- 比 `max(v_x, 0)` 更不容易奖励错误方向的高速运动。

潜在问题：

- 仅使用局部目标距离可能鼓励穿过障碍或切角。
- 噪声距离可能导致来回抖动获得奖励。

应使用滤波后的目标距离，并将 progress 与 collision/clearance constraint 联合。

### 2.4 Centering Reward

Centering reward 鼓励机身位于 passage 中心并保持左右 clearance 均衡：

```text
r_center =
  exp(-k_y * lateral_error^2)
  * exp(-k_c * (clearance_left - clearance_right)^2)
```

作用：

- 减少单侧机身碰撞。
- 为窄入口提供可解释的横向对齐信号。

潜在问题：

- 在 L/S-shaped corridor 中，几何中心不一定是最优轨迹。
- 过强 centering 会抑制绕障和必要的偏置进入。
- 若没有 progress gate，机器人可能停在入口中心获取奖励。

建议只在检测到 passage frame 时启用，并根据 corridor curvature 和 mode 调整权重。

### 2.5 Collision Cost

Collision cost 对机身、腿部或环境接触进行惩罚：

```text
c_collision =
  I(body_contact)
  + lambda_impulse * clipped_contact_impulse
```

作用：

- 防止策略以碰撞换取进度。
- 区分正常足地接触与机身/非期望腿部碰撞。

潜在问题：

- 只使用二值碰撞会产生稀疏梯度。
- 仿真碰撞检测或 allow-sliding 可能被策略利用。
- 若所有接触都惩罚，会破坏正常步态。

必须按 link 和 contact type 区分足端支撑与非期望碰撞，并独立记录 near-contact diagnostic。

### 2.6 Energy Cost

Energy cost 控制力矩、机械功或动作幅值：

```text
c_energy =
  sum_j |tau_j * qdot_j|
```

也可在训练早期使用归一化的 `sum_j tau_j^2`。

作用：

- 抑制高频、大力矩步态。
- 提高执行效率并减小实机热负荷。

潜在问题：

- 权重过大将产生低幅度、低速度甚至静止策略。
- 仿真中的机械功不一定等价于真实电池能耗。

应以任务成功和稳定性为前提报告能耗，不能单独以低能耗排序失败策略。

### 2.7 Smoothness Cost

Smoothness cost 惩罚相邻动作或关节目标突变：

```text
c_smooth =
  ||a_t - a_{t-1}||^2
  + lambda_jerk * ||a_t - 2a_{t-1} + a_{t-2}||^2
```

作用：

- 减少关节抖动和速度命令振荡。
- 缩小仿真策略与实机执行器带宽之间的差异。

潜在问题：

- 过强 smoothness 会抑制碰撞后的快速恢复。
- 若对高层模式切换也施加同一平滑项，Recover 可能无法及时反向。

建议按 mode 使用不同 smoothness 权重，并在真实紧急恢复时允许受限的快速响应。

### 2.8 Termination Cost

Termination cost 用于跌倒、严重碰撞、越界和不可恢复状态：

```text
c_termination =
  I(fall or severe_collision or invalid_state)
```

作用：

- 让策略区分可恢复的小扰动与必须终止的严重失败。
- 缩短明显失败 episode，提高训练效率。

潜在问题：

- 若 timeout 不属于 termination，策略可能学会等待。
- 若 termination penalty 过大，策略会极度保守或利用 simulator reset 条件。
- 若成功也触发统一 termination penalty，会抵消 success reward。

成功终止、失败终止和时间截断必须分别处理。

## 3. Why This Decomposition

四足窄通道运动同时包含任务目标、几何约束和动力学约束：

| Objective | Reward terms |
|---|---|
| 执行高层命令 | tracking |
| 保持稳定 | alive、termination |
| 完成导航 | forward |
| 保持通道姿态 | center |
| 避免危险接触 | collision |
| 提高实机可执行性 | energy、smooth |

该分解使每个失败来源都能通过独立 reward term 诊断。它也适合分层系统：高层 DEGNAV 负责 waypoint 和 mode，低层 RL 负责命令跟踪、接触稳定和运动效率。

不建议用同一个 reward 同时训练语言目标理解、通道 Reject 和关节控制。过长的 credit-assignment chain 会让低层策略通过速度或姿态行为补偿高层错误决策，削弱模块可解释性。

## 4. Potential Failure Modes

| Failure mode | Reward cause | Observable symptom |
|---|---|---|
| 原地站立 | alive/energy 权重过大 | 高回报、低 progress、timeout |
| 高速撞击 | tracking/forward 过大 | progress 高、collision 高 |
| 入口左右振荡 | center 与 heading 目标竞争 | `w_z` 频繁换向 |
| 过度保守 | collision/termination 过大 | 低速、false reject、停在入口 |
| 拒绝恢复 | smoothness 过大 | 碰撞后不倒退、不快速调整 |
| 沿墙擦行 | collision detector 不敏感 | nominal success 高、clearance proxy 低 |
| 循环刷 progress | 非势函数进度或噪声 | 前后往复仍累计正奖励 |
| 提前结束 | termination/reset 处理错误 | 策略主动触发 reset |
| 在开阔区等待 | clearance/alive bonus 无上限 | 不进入 passage 但回报持续增加 |

当前仓库的 `FairNarrowPassageRewardWrapper` 已针对最后一种问题进行修正：它裁剪开阔区 body-margin bonus，并对 outside-idle、timeout 和 no-progress 行为增加惩罚。该经验说明 reward 设计必须通过行为审计，而不能只看 episode return。

## 5. Avoiding Reward Hacking

### 5.1 Use Potential-based Progress

采用状态势函数差：

```text
r_progress = gamma * Phi(s_{t+1}) - Phi(s_t)
```

它可以减少往返运动重复获得 forward reward 的机会。

### 5.2 Separate Rewards from Evaluation Metrics

训练 reward 与论文指标不能完全相同。应保留策略不可直接优化的独立评测：

- Success / SPL。
- Collision 和 body contact。
- Timeout / stuck。
- Oscillation count。
- Foot slip 和 fall rate。
- Energy per successful meter。
- Recovery success。

若训练 return 提升但这些指标恶化，应判定为 reward exploitation。

### 5.3 Log Every Reward Term

每一步记录：

```text
reward/tracking
reward/alive
reward/forward
reward/center
reward/collision
reward/energy
reward/smooth
reward/termination
```

训练报告应展示 term-wise mean、episode sum 和与 success 的相关性。单一总 reward 曲线无法解释策略到底学会了什么。

### 5.4 Bound Dense Bonuses

- 对 clearance、alive 和 center 奖励设置上限。
- 只在 passage region 或有效任务阶段启用对应奖励。
- 让 success reward 大于通过拖延可累计的所有 dense bonus。
- 让 collision/termination cost 大于短时 shortcut progress。

### 5.5 Treat Safety as Constraints

对于碰撞、跌倒和力矩上限，推荐从纯 reward shaping 升级为约束优化：

```text
maximize  E[sum_t r_task]

subject to
  E[collision_cost] <= epsilon_collision
  E[fall_cost]      <= epsilon_fall
  E[torque_cost]    <= epsilon_torque
```

可使用 Lagrangian constrained RL、action projection 或 model-based safety filter。除非接触模型和真实机器人阈值经过标定，否则仍应称为 constraint-aware control，而不是物理安全保证。

### 5.6 Domain Randomization and Adversarial Tests

随机化摩擦、质量、质心、控制时延、深度噪声、外力和执行器增益。额外构造：

- 高 tracking reward 但前方有障碍。
- 可在中心等待但必须前进才能成功。
- 必须短暂倒退才能恢复。
- 低能耗路径更长但任务超时。

这些场景专门检查 reward loophole。

## 6. Improving Methodological Novelty

单纯增加 reward term 通常不足以构成论文创新。建议将 reward 与本文的 geometry、belief、memory 和 mode 贡献绑定。

### 6.1 Mode-conditioned Reward

```text
r_t = r_task + r_mode(m_t)
```

| Mode | Dominant objective |
|---|---|
| Commit | progress、tracking、clearance |
| Explore | information gain、低速、geometry uncertainty reduction |
| Recover | unstuck progress、safe reverse、realignment |
| Reject | correct rejection、avoid repeated infeasible commitment |

该设计避免用同一 forward reward 训练相反语义的 Commit 和 Recover。

### 6.2 Belief-conditioned Risk Budget

根据 feasibility belief 调整允许速度和风险：

```text
v_limit = f(p_feas, delta_var, memory_risk)
```

高不确定性时降低速度并奖励信息增益；低通过概率且历史失败充分时奖励正确 Reject。创新点应是 belief 如何改变决策和约束，而不是简单增加一个 risk penalty。

### 6.3 Morphology-aware Clearance Objective

将固定 clearance 替换为 yaw、机身尺寸和姿态相关的 required width：

```text
margin_t = d_hat_t - W_req_cons(theta_t, roll_t, morphology)
```

Reward 或 constraint 由该 margin 决定，使同一策略可适配不同机身包络。

### 6.4 Failure-memory Consistency

对历史相似失败通道重复 Commit 施加 wasted-attempt cost，同时对相似可行 passage 的 false reject 施加 interference cost：

```text
c_memory =
  I(repeated_infeasible_commit)
  + lambda_interference * I(false_reject_similar_feasible)
```

这比“有 memory 就减速”更能体现 failure-aware decision 的研究问题。

### 6.5 Recovery-specific Curriculum

Recovery 需要专用初始状态和监督，不能指望延长普通训练自动出现：

1. 人工初始化轻微卡滞和碰撞后状态。
2. 提供 safe reverse / re-alignment demonstrations。
3. 先训练 recovery skill，再联合 mode selector。
4. 在独立 recovery test set 上报告成功率。

## 7. Recommended Experimental Protocol

### 7.1 Ablations

1. Tracking + alive only。
2. + progress。
3. + centering / geometry。
4. + collision constraint。
5. + energy / smoothness。
6. Mode-conditioned reward。
7. Belief-conditioned risk。
8. Failure-memory consistency。

### 7.2 Metrics

每个版本至少报告 3-5 seeds：

- Task Success 和 SPL。
- Collision、fall、timeout 和 stuck。
- Oscillation count。
- Recovery Success。
- Tracking RMSE。
- Roll/pitch stability。
- Foot slip。
- Energy per successful meter。
- Reward term totals。

### 7.3 Current Evidence Boundary

当前仓库已经验证 narrow-passage navigation reward logging 和 direct-control PPO/SAC/TD3 训练路径，但尚无可核验的 Isaac Sim locomotion 多 seed 曲线、足端接触、能耗或 Lite3 sim-to-real 统计。因此，本文件中的 locomotion reward 设计是 Method proposal，不应写成已完成实验结论。

## 8. Paper-facing Formulation

推荐 Method 表述：

> We formulate quadruped narrow-passage control as a mode-conditioned locomotion problem. The reward separates command tracking and task progress from morphology-aware centering, undesired contact, actuation cost, and temporal smoothness. Potential-based progress and bounded dense bonuses reduce stationary and oscillatory reward exploits, while collision and termination are treated as explicit costs rather than implicit failure labels.

推荐创新性表述：

> The contribution is not the number of reward terms, but the coupling of feasibility belief, robot morphology, failure memory, and mode-specific locomotion objectives within a hierarchical navigation-control interface.
