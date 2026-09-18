# 最终方法状态表（MP3D，2026-09-05）

本页把官方 PointNav 基准与本文派生窄通道任务分开。两者机器人半径、动作
空间、滑动设置、Success 定义和 episode 集均不同，不能把数值放进同一个
排行榜。本文所有新表均不包含 strict-success 列。

## A. Habitat 官方 PointNav v1 / MP3D val

固定 495 个官方验证 episode；随机或采样策略使用 3 个评测种子，确定性策略
使用 1 个种子。所有方法/种子已经审计为完全相同的
`(scene_id, episode_id)` 集。

| 方法 | 官方定位 | 策略输入 → 输出 | Success | SPL | 到目标距离 | 有接触的 episode | 步数 | 状态/论文用途 |
|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| ForwardOnly | 官方示例下界 | PointGoal 距离 → Forward/Stop | 0.40% | 0.00 | 11.92 | 100.00% | 498.29 | 完成；附录下界 |
| Random Agent | 官方随机下界 | PointGoal 距离 → 随机离散动作/Stop | 2.49±0.34% | 0.01±0.00 | 11.28±0.05 | 100.00% | 493.79±0.58 | 完成；随机下界 |
| RandomForward | 官方弱反应式下界 | PointGoal 距离 → 80% Forward/随机转向/Stop | 1.41±0.29% | 0.01±0.00 | 12.04±0.10 | 100.00% | 495.27±0.45 | 完成；可放附录 |
| GoalFollower | 官方目标方向策略 | PointGoal 距离/方位 → 转向/Forward/Stop | 16.36% | 0.16 | 9.30 | 99.39% | 424.59 | 完成；反应式基线 |
| PointNav PPO | 官方学习基线 | RGB-D + PointGoal + GRU → 4 个离散动作 | 78.52±0.34% | 0.64±0.01 | 2.24±0.07 | 88.28±0.92% | 166.52±2.68 | 完成；主官方学习基线 |
| DD-PPO | 官方强循环学习基线 | Depth + PointGoal + LSTM → 4 个离散动作 | 93.87±0.42% | 0.85±0.00 | 1.24±0.03 | 97.17±0.87% | 117.84±0.89 | 完成；Gibson-2+→MP3D 迁移，须注明 |
| ShortestPathFollower | Habitat NavMesh Oracle | NavMesh + 目标位置 → 最短路径离散动作 | 100.00% | 1.00 | 0.08 | 0.61% | 75.19 | 完成；仅 Oracle 上界 |

“有接触的 episode”表示 episode 内至少发生一次模拟器碰撞，不是逐步碰撞率。
原始碰撞次数保存在各方法的 `summary.csv` 中。

## B. MP3D 派生窄通道验证

训练/验证场景遵循官方 MP3D train/val 场景划分且场景交集为 0；验证集为相同
80 个派生 episode（64 个可行、16 个 false-feasible 候选），机器人直径
0.36 m、关闭 sliding、连续速度控制。该协议不是 Habitat 官方 PointNav 榜单。

| 方法 | 部署输入 → 输出 | 种子 | All Success | Feasible Success | Correct Reject | Collision | 平均步数 | 当前结论 |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| Geometry-19D / no-memory | 深度派生 19D → 模式/连续控制 | 1 | 82.50% | 98.44% | 0.00% | 0.00% | 136.03 | 完成；无拒绝能力 |
| episodic kNN memory-19D | 19D + kNN 失败记忆 → 模式/连续控制 | 1 | 82.50% | 98.44% | 0.00% | 0.00% | 136.03 | 完成；与无记忆完全相同 |
| DEGNAV memory-19D | 19D + DEGNAV 失败记忆 → 模式/连续控制 | 1 | 82.50% | 98.44% | 0.00% | 0.00% | 136.03 | 完成；当前数据不支持记忆增益 |
| DEGNAV-E2E (Depth, offline BC) | Raw Depth + PointGoal + 前一动作/结果 + GRU → 四模式/连续控制 | 3 | 83.33±0.59% | 99.48±0.74% | 79.17±2.95% | 0.00±0.00% | 63.55±3.57 | 完成；Decision accuracy 95.42±1.18%，False Reject 2.08±0.74% |

派生表中的 `false-feasible` 是生成器候选标签，不是经接触仿真证明的绝对不可行
真值；应同时报告 traversal、timeout 和显式 Reject，不能把失败自动算作拒绝。

相对三个完全相同的 19D 行，E2E 的 All Success 增加 0.83 个百分点、Feasible
Success 增加 1.04 个百分点、Decision accuracy 增加 16.67 个百分点、Correct
Reject 增加 79.17 个百分点，平均步数减少 72.48（53.28%）。代价是 2.08%
False Reject；而且 240 个种子-episode 中只有 1 个实际使用 Recover，因此当前
证据支持进入、探索和拒绝，不支持“恢复模块已经可靠”。E2E 在这里表示视觉到
高层模式，连续速度仍由固定低层控制器生成，并非单体像素到速度策略。

## C. 不能并入上述 MP3D 正式表的现有结果

以下结果来自 Procedural-v2 的 19D 分析型局部导航环境，不是官方 Habitat
PointNav，也不是原始图像端到端评测，只能作为独立诊断表引用。

| 方法 | Success | Collision | Timeout/stuck | 解释 |
|:---|---:|---:|---:|:---|
| Direct-control PPO | 84.3±0.5% | 15.7±0.5% | 0.0±0.0% | 程序化 19D 连续控制基线 |
| Recurrent PPO（from scratch） | 0.1±0.1% | 20.3±0.3% | 79.7±0.4% | 从零训练发生低进展/超时塌缩 |
| Recurrent PPO（Direct-init + fine-tune） | 85.0±0.9% | 15.0±0.9% | 0.0±0.0% | 超过 50% 门槛，但不是独立从零基线 |
| DEGNAV + geometry-guided memory | 83.1±0.2% | 0.0±0.0% | 16.9±0.2% | 零接触不等于完整安全，near-collision 为 45.1% |

## 原始证据入口

- 官方 PointNav：`../official_pointnav_mp3d_v1_20260904/`
- 派生 19D / E2E：`../degnav_e2e_mp3d_narrow_v1_20260904/`
- 环境快照：`environment_snapshot.md`
