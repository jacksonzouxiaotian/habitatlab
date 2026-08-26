# 窄通道局部导航统一基线实验与代码工作报告

日期：2026-08-26

代码分支：`narrow-passage-rl-memory`

核心代码与结果提交：`27d044066ed06c42b8ab7c2367954a307d7fad9a`

## 1. 执行摘要

本阶段完成了 Geometry rule、Direct-control PPO、原始 Recurrent PPO、
Direct-init Recurrent PPO 和 DEGNAV geometry-guided memory 的统一配对评测。
所有方法使用相同的 Procedural-v2 场景顺序、环境种子、通道几何、宽度范围
和最大步数。最终数据包含 5 个方法、3 个评测种子、每个方法/种子 500 个
episode，共 7,500 条逐 episode 记录。

主要结果如下：

- Direct-control PPO 是当前最可靠的独立学习基线，成功率为
  `84.3±0.5%`。
- 原始 from-scratch Recurrent PPO 训练失败，成功率仅 `0.1±0.1%`，
  `79.7±0.4%` 的 episode 超时或卡住。
- 经过 Direct PPO 行为初始化和低学习率 RecurrentPPO 微调后，更新版
  Recurrent PPO 达到 `85.0±0.9%`，超过预设的 50% 成功率门槛。
- 更新版 Recurrent PPO 不是独立于 Direct PPO 的 from-scratch baseline，
  因而不能用它证明“循环记忆本身优于前馈策略”。
- DEGNAV 的名义成功率为 `83.1±0.2%`，模拟器碰撞记录为 0，但近碰撞率
  `45.1±2.2%`、超时/卡住率 `16.9±0.2%`，且本轮没有产生正确拒绝。
- 当前统一基线仍是 19 维解析几何观测上的局部导航实验，不是 Habitat RGB
  端到端导航结果。

## 2. 工作目标

本阶段回答以下问题：

1. 在大量不同通道形状和宽度上，反应式规则、前馈 RL、循环 RL 和
   几何记忆方法的进入与通过能力有何差异？
2. 原始 Recurrent PPO 为什么显著弱于 Direct-control PPO？
3. 单纯增加 Recurrent PPO 训练步数能否解决问题？
4. 是否能够得到至少 50% held-out 成功率的 Recurrent PPO checkpoint？
5. 现有 DEGNAV 结果是否已经证明探索、恢复和拒绝能力？
6. 如何把训练、评测、逐 episode 数据、checkpoint provenance 和论文表格
   整理成可复核的统一实验产物？

## 3. 实验边界

### 3.1 当前实验是什么

当前实验是二维 Procedural-v2 窄通道局部控制基准。环境提供机器人相对通道
几何和局部障碍的解析观测，策略直接输出连续速度命令：

```text
19-D local observation
        |
        +--> rule / feed-forward PPO / recurrent PPO / geometry FSM
        |
        +--> action = [linear velocity vx, angular velocity wz]
        |
        +--> OBB-aware procedural dynamics and termination
```

### 3.2 当前实验不是什么

- 不是原始 RGB 图像直接输入的 Habitat PointNav。
- 不是端到端 VLN/VLA 导航。
- 不是多训练种子神经网络稳定性研究。
- 不是独立的 episodic kNN memory baseline。
- 不是 `no-memory`、`no-uncertainty`、`no-yaw-prior` 的完整因果消融表。
- 不是能够证明真实机器人安全性的部署实验。

因此，本报告中的结论只适用于当前 Procedural-v2 局部导航协议。

## 4. 环境、观测和动作

### 4.1 通道类型

统一评测覆盖七类通道：

1. `straight`：直通道；
2. `l_shaped`：单个 90 度转弯；
3. `s_shaped`：两个方向相反的转弯；
4. `narrow_exit`：入口较宽、出口收窄；
5. `narrow_entry`：入口较窄、内部变宽；
6. `asymmetric`：单侧突出或左右不对称；
7. `false_feasible`：入口看似可行、内部实际不可通行。

通道宽度在 `[0.45, 0.90] m` 内采样，单 episode 最大 400 步。

### 4.2 19 维观测

观测包含：

- 左、中、右三个近距离射线深度；
- 左、中、右三个远距离射线深度；
- 左右本体净空；
- 估计通道宽度；
- 最小本体余量；
- 航向误差；
- 横向偏移；
- 到目标距离；
- 当前/历史动作；
- 卡住分数；
- 碰撞标记。

这一观测已经包含当前位置的主要几何状态，因此在当前环境中近似 Markov。
这也是前馈 Direct PPO 容易优化、而大规模 LSTM 没有获得明显信息增益的重要
原因。

### 4.3 动作

所有控制方法最终输出同一连续动作接口：

```text
[vx, wz] = [forward/backward velocity, angular velocity]
```

## 5. 方法实现对比

| 方法 | 学习方式 | episode 内状态 | 跨 episode 记忆 | 显式模式 | 主要定位 |
|:---|:---|:---:|:---:|:---|:---|
| Geometry rule | 无训练 | 否 | 否 | 否 | 最低能力反应式基线 |
| Direct-control PPO | PPO | 否 | 否 | 否 | 独立前馈学习基线 |
| Recurrent PPO | PPO | LSTM | 否 | 否 | from-scratch 循环基线 |
| Recurrent PPO Direct-init | 行为蒸馏 + PPO | LSTM | 否 | 否 | 修复后的循环策略 |
| DEGNAV memory | 几何规则/FSM | 控制器状态 | 是 | 是 | 可解释几何与失败历史方法 |

### 5.1 Geometry rule

规则为固定前进速度加航向误差比例转向。它不进行训练，也不利用历史状态。
优点是实现简单、可解释、计算量小；缺点是不能有效预测转角、非对称空间和
入口后方的结构变化。

### 5.2 Direct-control PPO

Direct PPO 使用两层前馈 MLP 将当前 19 维观测直接映射为 `[vx, wz]`。
策略共有 11,077 个参数，使用 fair reward、8 个并行环境和约 3.006M 次环境
交互训练。它没有显式 Commit/Explore/Recover/Reject 语义，也不会保存跨
episode 历史。

### 5.3 原始 Recurrent PPO

原始模型使用独立的 256 维 actor LSTM 和 critic LSTM，共 608,709 个参数，
约为 Direct PPO 的 55 倍。LSTM 状态只在单个 episode 内保留，reset 时清零，
因此它不是跨 episode memory。

### 5.4 更新版 Recurrent PPO

更新模型保留 RecurrentPPO 标准 actor 接口，但将架构改为：

- 64 维单层 actor LSTM；
- 前馈 critic；
- 总参数量 39,877；
- Direct PPO 行为初始化；
- 低学习率 on-policy PPO 微调。

它仍然具有 episode 内循环状态，但性能初始化来自 Direct PPO 教师。

### 5.5 DEGNAV geometry-guided memory

DEGNAV 由几何可行性估计、显式有限状态控制器、转角跟随、恢复逻辑和
CrossEpisodeMemory 组成。典型控制模式包括：

```text
APPROACH -> ALIGN -> ENTER -> TRAVERSE -> EXIT
                    |             |
                    +-> EXPLORE   +-> RECOVER
```

跨 episode memory 保存可审计的成功、几何失败、控制失败和 censored outcome，
使用连续、形态感知的相似度进行检索，并以有界证据修正控制器的谨慎程度。
记忆在每个评测种子边界清空，不跨评测种子泄漏。

## 6. 训练过程

### 6.1 共同 RL 配置

| 配置 | 值 |
|:---|---:|
| 训练种子 | 0 |
| 并行环境 | 8 |
| rollout steps / env | 1024 |
| minibatch | 256 |
| 折扣因子 | 0.99 |
| GAE lambda | 0.95 |
| PPO clip | 0.2 |
| reward | fair reward wrapper |
| 通道类型 | 全部七类 |

fair reward 对前进进度、成功和安全净空给出正反馈，对碰撞、超时、卡住、
横向偏移、航向误差、动作突变和通道外停滞给出惩罚。其目的之一是消除原生
clearance reward 在开放区域过大的停留激励。

### 6.2 Direct PPO

- 训练设备：CPU；
- PPO epochs：3；
- checkpoint 内部步数：3,006,464；
- 正式成功率：84.3±0.5%。

### 6.3 原始 Recurrent PPO

- 训练设备：CUDA；
- PPO epochs：3；
- checkpoint 内部步数：3,006,464；
- 0.5M–3M 训练回报始终没有进入稳定正回报区；
- episode 长度后期上升，表明策略趋向低推进/停滞。

另一个历史 3M、10 epochs 的 Recurrent PPO 结果也只有约 13%，说明问题不
只是本轮 epochs 从 10 降到 3。

### 6.4 轻量 from-scratch Recurrent 诊断

为了检查模型规模是否是唯一原因，训练了 64 维 actor LSTM + 前馈 critic、
10 epochs 的候选模型。该模型在早期具有更好的训练回报和更高采样速度，但
在 0.5M checkpoint 的 100 个 held-out episode 上结果为：

| 指标 | 结果 |
|:---|---:|
| Success | 0% |
| Collision | 24% |
| Timeout | 76% |
| Avg steps | 368.25 |

因此，缩小网络和提高样本利用率并没有单独解决 recurrent actor 的信用分配
与停滞局部最优问题。

### 6.5 Direct PPO 行为初始化

最终采用以下修复过程：

1. 使用 Direct PPO 在 300 个与正式评测隔离的训练 episode 上收集观测；
2. 教师在该数据集上的成功率为 83.0%；
3. 使用长度 32 的序列窗口监督 recurrent actor 的动作分布均值；
4. 动作均值 MSE 从 1.848 降到约 0.002；
5. 使用学习率 `1e-5`、3 epochs 继续 106,496 次 RecurrentPPO 交互；
6. checkpoint 累计步数为 606,496；
7. 100-episode held-out 门槛测试达到 90%；
8. 通过门槛后才启动完整三种子评测。

微调期间训练回报保持约 `+31` 到 `+39`，episode 长度约 77–87，KL 通常在
0.001–0.003，未再次出现 actor 快速退化。

## 7. 正式评测协议

| 项目 | 设置 |
|:---|:---|
| 环境 | `HarderNarrowPassageEnv` / Procedural-v2 |
| 评测种子 | 42、43、44 |
| 每方法每种子 episode | 500 |
| 每方法总 episode | 1,500 |
| 方法数 | 5 |
| 总记录数 | 7,500 |
| 通道宽度 | 0.45–0.90 m |
| 最大步数 | 400 |
| 推理 | deterministic |
| 场景配对 | seed、类型、宽度、可通行标签、顺序完全一致 |

表中 `mean±std` 是三个评测场景种子的均值和总体标准差，不是多个独立 RL
训练种子的均值和标准差。

## 8. 正式结果

### 8.1 总体结果

| 方法 | Success ↑ | Collision ↓ | Near collision ↓ | Timeout/stuck ↓ | Correct reject ↑ | False reject ↓ | Avg steps ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Geometry rule | 23.3±1.9% | 70.3±2.2% | 38.6±1.6% | 6.3±0.5% | — | — | 60.0±2.6 |
| Direct-control PPO | 84.3±0.5% | 15.7±0.5% | 14.9±0.4% | 0.0±0.0% | — | — | 68.5±0.1 |
| Recurrent PPO | 0.1±0.1% | 20.3±0.3% | 19.1±0.2% | 79.7±0.4% | — | — | 323.8±1.1 |
| Recurrent PPO Direct-init | **85.0±0.9%** | **15.0±0.9%** | **14.1±0.7%** | **0.0±0.0%** | — | — | **68.3±0.2** |
| DEGNAV memory | 83.1±0.2% | 0.0±0.0% | 45.1±2.2% | 16.9±0.2% | 0.0±0.0% | 0.0±0.0% | 188.5±1.7 |

平均步数必须结合终止原因解释。Geometry rule 的 60 步不是最高效率的证明，
因为大量 episode 很快碰撞终止；原始 Recurrent PPO 的 323.8 步则直接反映
大规模超时。

### 8.2 分评测种子的更新版 Recurrent PPO

| 评测种子 | Success | Collision | Timeout/stuck | Avg steps |
|:---|---:|---:|---:|---:|
| 42 | 84.2% | 15.8% | 0.0% | 68.44 |
| 43 | 84.6% | 15.4% | 0.0% | 67.95 |
| 44 | 86.2% | 13.8% | 0.0% | 68.45 |

三个种子均显著高于 50% 工程门槛，没有依赖单一评测种子的偶然结果。

### 8.3 分通道成功率

| 方法 | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False feasible |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Geometry rule | 59.6±0.3% | 1.8±0.6% | 0.9±0.6% | 29.0±1.3% | 34.1±2.7% | 29.8±5.6% | 0.0±0.0% |
| Direct PPO | 96.7±1.1% | 100.0±0.0% | 100.0±0.0% | 97.1±1.0% | 91.7±1.5% | 61.8±4.6% | 0.0±0.0% |
| Recurrent PPO | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 0.6±0.8% | 0.0±0.0% | 0.0±0.0% |
| Recurrent PPO Direct-init | 98.4±1.2% | 100.0±0.0% | 100.0±0.0% | 97.0±0.3% | 93.8±3.1% | 63.8±4.5% | 0.0±0.0% |
| DEGNAV memory | 100.0±0.0% | 100.0±0.0% | 98.9±0.9% | 100.0±0.0% | 99.4±0.9% | 32.9±8.7% | 0.0±0.0% |

`false_feasible` 的目标成功率为 0 并不等价于正确拒绝。正确行为应由
`correct_reject` 单独衡量；当前 DEGNAV 的 correct reject 仍为 0，因此这些
episode 主要是尝试后失败，而不是提前做出正确拒绝。

## 9. 结果解释

### 9.1 Geometry rule

Geometry rule 在直通道尚有 59.6% 成功率，但 L/S 通道接近完全失败，且总体
碰撞率达到 70.3%。它验证了单步航向比例控制无法处理转角承诺、非对称空间和
后续通道变化。

### 9.2 Direct PPO

Direct PPO 在 L/S 通道达到 100%，说明当前 19 维观测足以让前馈策略学习局部
转弯行为。其主要残余失败集中在 asymmetric 和部分 narrow-entry 场景。它是
当前可用于论文公平比较的最强独立学习基线。

### 9.3 原始 Recurrent PPO

原始 Recurrent PPO 几乎所有通道类型均为 0%，失败集中表现为长时间低推进和
超时，而不是单纯高碰撞。训练曲线、历史 10-epoch checkpoint 和轻量模型诊断
共同排除了“只需要继续原配置训练”这一解释。

### 9.4 更新版 Recurrent PPO

更新模型恢复到与 Direct PPO 几乎相同的行为分布，并略微提高 straight、
narrow-entry 和 asymmetric 的点估计。但由于它使用 Direct 教师、只有一个训练
种子，而且 0.7 个百分点的总体差异很小，不能宣称它在统计意义上优于 Direct
PPO。其有效结论是：循环策略模块在正确初始化后能够达到并稳定保持高成功率。

### 9.5 DEGNAV memory

DEGNAV 在多数规则通道上表现强，但 asymmetric 只有 32.9%，并且平均用时远高
于 PPO。模拟器碰撞为 0 是积极信号，但 45.1% near-collision 表明许多 episode
进入小于 5 cm 的风险余量；当前环境还允许某些滑动/接触行为，因此不能仅依据
collision 字段宣称完整安全。

本轮 `fsm_cross_memory` 主要执行标准/谨慎记忆路径，终端 memory reject 没有在
统一评测中生效。因而现有数据不能支持“DEGNAV 已经做出更好的拒绝决策”。

## 10. Recurrent PPO 失败原因证据链

目前证据支持以下因果链：

```text
近似 Markov 的 19-D 观测
        +
默认双 256-D LSTM（参数量约 55 倍）
        +
更困难的序列 minibatch / credit assignment
        |
        v
actor 更新进入低推进动作盆地
        |
        v
episode 长度上升、训练回报长期为负
        |
        v
held-out 中 79.7% timeout/stuck
```

支持证据包括：

1. Direct PPO 在相同观测和 reward 下能够达到 84.3%；
2. 原始 Recurrent 0.5M–3M 始终没有形成可靠正回报；
3. 历史 10 epochs、3M Recurrent 仍只有约 13%；
4. 缩小 LSTM 后 0.5M held-out 仍为 0%；
5. 同一轻量 LSTM 经教师初始化后立即达到 90% 门槛成功率；
6. 低学习率 PPO 微调后完整评测仍为 85.0%。

这说明观测和动作接口本身可学习，关键困难在 recurrent actor 的优化初始状态
和训练动力学。

## 11. 代码工作

### 11.1 训练脚本

[`train_sb3_v2.py`](../train_sb3_v2.py) 新增/统一了：

- 并行环境数量；
- Dummy/Subproc vector environment；
- PPO rollout、batch、epochs 和 device 参数；
- fair/native reward 切换。

[`train_recurrent_ppo_v2.py`](../train_recurrent_ppo_v2.py) 新增：

- 正确的累计步数续训语义；
- 可配置学习率和 entropy coefficient；
- LSTM hidden size/layers；
- separate/shared/feed-forward critic 模式；
- 并行环境和 checkpoint 配置。

### 11.2 教师初始化

[`distill_direct_to_recurrent.py`](../distill_direct_to_recurrent.py) 实现：

- 隔离训练种子的教师轨迹采集；
- sequence-major LSTM 训练窗口；
- Direct PPO 与 Recurrent PPO 动作分布均值拟合；
- actor-only 优化和梯度裁剪；
- 可保存后续 PPO 可继续加载的标准 RecurrentPPO checkpoint。

### 11.3 统一评测

[`eval_unified_local_baselines.py`](../eval_unified_local_baselines.py) 实现：

- Geometry、Direct PPO、Recurrent PPO、Direct-init Recurrent 和 DEGNAV 的
  统一方法 ID；
- 完全一致的 episode seed 和 scenario ID；
- 训练种子与评测种子分离；
- 分片评测的 `episode-start-index`；
- 多个逐 episode CSV 的合并；
- per-seed 和跨 seed 汇总；
- checkpoint SHA256、训练步数和 Git provenance；
- 不覆盖旧结果的版本化输出目录。

### 11.4 环境与几何依赖

本轮提交同时保存当前实验实际依赖的：

- [`procedural_env_v2.py`](../procedural_env_v2.py)；
- [`eval_harder_benchmark.py`](../eval_harder_benchmark.py)；
- [`cross_episode_memory.py`](../cross_episode_memory.py)；
- [`robot_morphology.py`](../narrow_passage/models/robot_morphology.py)；
- [`dynamic_feasibility.py`](../narrow_passage/models/dynamic_feasibility.py)；
- [`feasibility_ablation.py`](../narrow_passage/models/feasibility_ablation.py)。

这样 GitHub 上的统一评测脚本不会依赖只存在于本地工作区的未提交 API。

## 12. 结果与 checkpoint provenance

### 12.1 正式结果文件

目录：

```text
examples/narrow_passage_rl/results/narrow_passage_rl/
  unified_local_baselines_recurrent_updated_20260824/
```

文件说明：

| 文件 | 内容 |
|:---|:---|
| `episodes.csv` | 7,500 条逐 episode 原始结果 |
| `summary_by_eval_seed.csv` | 15 个方法/种子汇总行 |
| `summary.csv` | 5 个方法的跨评测种子结果 |
| `paper_table_unified_local_baselines.md` | 当前主对比表 |
| `metadata.json` | 协议、路径、步数、SHA256 和版本信息 |
| `recurrent_ppo_retraining_report.md` | Recurrent 专项诊断摘要 |

### 12.2 checkpoint

checkpoint 目录受项目 `.gitignore` 管理，权重保留在本地而未强制上传 GitHub。
其审计信息如下：

| 模型 | 内部步数 | SHA256 |
|:---|---:|:---|
| Direct PPO | 3,006,464 | `c823a235f34b07447f184d5ac5b8c9a7b9c82da637f1a9346af0013aadd1dd48` |
| 原始 Recurrent PPO | 3,006,464 | `4af8284084b11629000b4703ae72556d0ee777eaa08c1dd4d038719075f1f792` |
| Direct-init Recurrent PPO | 606,496 | `e375573bd9b9550f2f3a37527dfeefc554e3680b80f3a93d5b2a54f5bec952d0` |

## 13. 复现命令

以下命令从仓库根目录执行。

### 13.1 Direct PPO

```bash
conda run -n habitat python examples/narrow_passage_rl/train_sb3_v2.py \
  --algo ppo --total-steps 3000000 --seed 0 --ctypes full \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/direct_reproduction \
  --eval-episodes 0 --n-steps 1024 --batch-size 256 --n-epochs 3 \
  --device cpu --num-envs 8 --vec-env subproc --reward-mode fair
```

### 13.2 原始 Recurrent PPO

```bash
conda run -n habitat python examples/narrow_passage_rl/train_recurrent_ppo_v2.py \
  --total-steps 3000000 --seed 0 --ctypes full \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_reproduction \
  --eval-episodes 0 --n-steps 1024 --batch-size 256 --n-epochs 3 \
  --lstm-hidden-size 256 --recurrent-value-mode separate \
  --device cuda --num-envs 8 --vec-env subproc --reward-mode fair
```

### 13.3 轻量 Recurrent 预训练阶段

```bash
conda run -n habitat python examples/narrow_passage_rl/train_recurrent_ppo_v2.py \
  --total-steps 500000 --seed 0 --ctypes full \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_lstm64_stage1 \
  --eval-episodes 0 --n-steps 1024 --batch-size 256 --n-epochs 10 \
  --learning-rate 0.00025 --ent-coef 0.01 \
  --lstm-hidden-size 64 --recurrent-value-mode feedforward \
  --device cuda --num-envs 8 --vec-env subproc --reward-mode fair
```

### 13.4 Direct PPO 行为初始化

```bash
conda run -n habitat python examples/narrow_passage_rl/distill_direct_to_recurrent.py \
  --teacher-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/unified_direct_ppo_seed0_3m_current/ppo_narrow_passage_v2.zip \
  --student-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_lstm64_stage1/recurrent_ppo_narrow_passage_v2.zip \
  --output-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_direct_init/recurrent_ppo_distilled \
  --episodes 300 --updates 1500 --batch-size 16 --sequence-length 32 \
  --learning-rate 0.001 --seed 1701 --device cuda
```

### 13.5 RecurrentPPO 保守微调

```bash
conda run -n habitat python examples/narrow_passage_rl/train_recurrent_ppo_v2.py \
  --load-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_direct_init/recurrent_ppo_distilled.zip \
  --total-steps 100000 --seed 0 --ctypes full \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_direct_init_finetuned \
  --eval-episodes 0 --n-steps 1024 --batch-size 256 --n-epochs 3 \
  --learning-rate 0.00001 --ent-coef 0.001 \
  --device cuda --num-envs 8 --vec-env subproc --reward-mode fair
```

### 13.6 统一评测

单次运行可以通过 `--methods` 选择方法。原始 Recurrent 和 Direct-init
Recurrent 使用不同 checkpoint，应分别评测后合并 episode CSV：

```bash
conda run -n habitat python examples/narrow_passage_rl/eval_unified_local_baselines.py \
  --direct-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/unified_direct_ppo_seed0_3m_current/ppo_narrow_passage_v2.zip \
  --recurrent-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/unified_recurrent_ppo_seed0_3m_cuda_current/recurrent_ppo_narrow_passage_v2.zip \
  --episodes 500 --eval-seeds 42 43 44 \
  --methods geometry_rule direct_control_ppo recurrent_ppo degnav_memory \
  --output-dir examples/narrow_passage_rl/results/narrow_passage_rl/unified_original_reproduction

conda run -n habitat python examples/narrow_passage_rl/eval_unified_local_baselines.py \
  --direct-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/unified_direct_ppo_seed0_3m_current/ppo_narrow_passage_v2.zip \
  --recurrent-model examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_direct_init_finetuned/recurrent_ppo_narrow_passage_v2.zip \
  --episodes 500 --eval-seeds 42 43 44 \
  --methods recurrent_ppo_direct_init \
  --output-dir examples/narrow_passage_rl/results/narrow_passage_rl/unified_direct_init_reproduction
```

长时间 recurrent 评测可以用 `--episode-start-index` 切成不重叠分片，再通过
`--episode-inputs` 合并。合并器会重新计算 per-seed 和总体表格。

## 14. 验证与审计

已完成以下检查：

- 7,500 条 episode；
- 15 个方法/评测种子组，每组恰好 500 条；
- 五个方法的 scenario ID、episode seed、通道类型、可通行标签和宽度完全配对；
- 7,500 个 episode ID 唯一；
- 汇总关键数值均为有限值；
- checkpoint 内部训练步数与 metadata 一致；
- 三个 checkpoint SHA256 重新计算一致；
- 所有新增/修改 Python 文件通过 `py_compile`；
- 从 Git 暂存快照完成模块导入和两回合规则基线运行；
- 暂存内容敏感凭据扫描无命中。

当前 `habitat` 环境没有安装 `pytest`，所以本轮没有声称执行 pytest 测试套件。
正式 7,500-episode 运行和快照 smoke test 是当前主要的运行验证证据。

## 15. 有效性威胁与不能过度声明的结论

### 15.1 只有一个 RL 训练种子

当前 `±` 反映评测场景种子变化，不反映神经策略从不同随机初始化训练的方差。
论文若要比较 Direct 和 Recurrent 的算法稳定性，应至少补三个独立训练种子。

### 15.2 Direct-init 的教师依赖

更新 Recurrent 使用 Direct PPO 行为初始化。它证明 RecurrentPPO 模块可以被
训练成有效控制器，但不是 recurrence 独立带来性能提升的证据。

### 15.3 DEGNAV memory 尚未被单独隔离

当前表没有使用完全相同控制器的 `no-memory` 和独立 episodic kNN 对照，因此
83.1% 不能归因于 cross-episode memory。控制器几何修复、转角逻辑和记忆同时
存在。

### 15.4 Reject 尚未验证

DEGNAV 在 false-feasible episode 上没有产生正确拒绝。本轮不能支持更优 Reject
决策的论文表述。需要把 terminal memory correction 接入统一评测的正式 selector，
并以 correct reject、false reject、time-to-reject 和 wasted attempt 为主要指标。

### 15.5 Procedural-to-Habitat 差距

解析射线和几何量不等于 RGB/D/semantic perception。传感器噪声、遮挡、尺度误差、
相机外参、Habitat 碰撞滑动和真实机器人动力学尚未被这一表格覆盖。

### 15.6 Collision 字段不是完整安全证明

DEGNAV 的 simulator collision 为 0，但 near-collision 较高，而且环境中的滑动或
接触语义可能允许风险轨迹继续到达目标。因此应同时保留 body margin、接触、
轨迹投影和真实机器人安全指标。

## 16. 后续工作优先级

### P0：补齐公平训练种子

- Direct PPO：至少 3 个训练种子；
- from-scratch Recurrent PPO：至少 3 个训练种子；
- Direct-init Recurrent PPO：至少 3 个学生初始化/微调种子；
- 用训练种子作为主要统计单位，评测场景作为配对重复测量。

### P0：修复并验证 Reject

- 将 `CrossEpisodeMemory.correct_feasibility()` 接入统一 selector；
- 为 `false_feasible` 增加专门门槛；
- 报告 correct reject、false reject、time-to-reject 和 wasted attempt；
- 检查拒绝是否依赖不可用的 ground-truth 通道标签。

### P1：隔离 memory 贡献

至少增加：

1. geometry controller、no memory；
2. geometry controller、episodic kNN memory；
3. geometry controller、DEGNAV memory；
4. DEGNAV no-uncertainty；
5. DEGNAV no-yaw-prior；
6. memory history shuffled/cross-scene interference。

### P1：证明 recurrence 是否真的被使用

- 推理时将 LSTM state 每步清零；
- 对历史顺序做 shuffle；
- 比较真实 hidden state、zero hidden 和前馈 student；
- 评估遮挡、延迟、观测 dropout 等真正部分可观测条件。

### P1：安全指标修复

- 将 OBB body overlap 设为明确终止或单独失败；
- 统一 simulator collision、swept-OBB clearance 和 body margin；
- 对 near-collision 阈值做灵敏度分析；
- 在 Habitat allow-sliding 开/关条件下分别评估。

### P2：接入 Habitat RGB-D

- RGB encoder 提供语义和局部目标；
- depth/geometry adapter 产生与 19 维接口一致的可审计量；
- 比较 oracle geometry、predicted geometry 和 RGB-only；
- 最终报告 Habitat SR/SPL、碰撞、恢复、拒绝和计算延迟。

## 17. 最终结论

当前最稳健的独立学习结论是 Direct-control PPO 在 Procedural-v2 19 维局部
观测上达到约 84% 成功率。原始 Recurrent PPO 并没有因为拥有 LSTM 而获得
优势，反而因网络规模和训练动力学进入停滞局部最优。通过 Direct 行为初始化、
轻量 LSTM 和保守 PPO 微调，可以将 Recurrent PPO 恢复到 85%，满足 50% 工程
门槛，但这是一种 teacher-assisted 修复结果。

DEGNAV 展现了强通道通过能力和零模拟器碰撞记录，但目前仍存在速度慢、近碰撞
高、非对称通道弱和终端拒绝未生效的问题。下一阶段最重要的工作不是继续增加
单个 checkpoint 的步数，而是补多训练种子、隔离 memory/recurrence 因果贡献、
修复 Reject 路径，并将解析几何基准迁移到 Habitat RGB-D 和真实机器人条件。
