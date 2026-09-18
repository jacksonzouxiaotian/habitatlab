# Habitat / MP3D 正式实验工作报告

日期：2026-09-05（Asia/Shanghai）

## 1. 执行结论

本轮先完成 Habitat 官方 PointNav 基线，再完成 MP3D 派生窄通道的 19D
局部方法和原始深度 E2E 方法。官方协议与派生协议在机器人尺寸、动作空间、
sliding、Success 定义和 episode 集上不同，因此报告中始终分表，禁止跨表直接
排序。所有新论文表均不提供 strict-success 列。

官方 MP3D PointNav 结果已经完整：PointNav PPO 为 `78.52±0.34%` Success、
`0.64±0.01` SPL；DD-PPO 为 `93.87±0.42%` Success、`0.85±0.00`
SPL，但 DD-PPO 权重训练于 Gibson-2+，必须标成跨域迁移。ShortestPathFollower
达到 100%，只作为使用 NavMesh 的 Oracle 上界。

在相同 80 个 MP3D 派生窄通道验证 episode 上，Geometry-19D、kNN-memory-19D
和 DEGNAV-memory-19D 的结果完全相同：All Success `82.50%`、Feasible
Success `98.44%`、Correct Reject `0%`。当前数据不支持“记忆提升性能”的
结论。

修复逐帧历史后，DEGNAV-E2E 的三种子闭环结果为：All Success
`83.33±0.59%`、Feasible Success `99.48±0.74%`、Decision accuracy
`95.42±1.18%`、候选不可行 Correct Reject `79.17±2.95%`、Feasible False
Reject `2.08±0.74%`、Collision `0.00±0.00%`、平均 `63.55±3.57` 步。
此前产生的 E2E 数字均已归档为无效诊断，不参与这些均值。

## 2. 正式协议 A：Habitat 官方 PointNav

数据为官方 MP3D PointNav v1 `val`，共 495 个 episode。配置保持官方
PointNav 默认值：agent radius 0.10 m、forward step 0.25 m、turn angle 10°、
sliding 开启、success distance 0.20 m、最大 500 步。只额外记录 collision 和
step count，不修改 Success/SPL 定义。

| 方法 | Episodes/seed | Seeds | Success | SPL | Distance to goal | 有接触的 episode | Steps |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ForwardOnly | 495 | 1 | 0.40% | 0.00 | 11.92 | 100.00% | 498.29 |
| Random Agent | 495 | 3 | 2.49±0.34% | 0.01±0.00 | 11.28±0.05 | 100.00% | 493.79±0.58 |
| RandomForward | 495 | 3 | 1.41±0.29% | 0.01±0.00 | 12.04±0.10 | 100.00% | 495.27±0.45 |
| GoalFollower | 495 | 1 | 16.36% | 0.16 | 9.30 | 99.39% | 424.59 |
| PointNav PPO | 495 | 3 | 78.52±0.34% | 0.64±0.01 | 2.24±0.07 | 88.28±0.92% | 166.52±2.68 |
| DD-PPO | 495 | 3 | 93.87±0.42% | 0.85±0.00 | 1.24±0.03 | 97.17±0.87% | 117.84±0.89 |
| ShortestPathFollower | 495 | 1 | 100.00% | 1.00 | 0.08 | 0.61% | 75.19 |

PointNav PPO 使用发布的 MP3D RGB-D ResNet18-GRU 权重；DD-PPO 使用发布的
Gibson-2+ Depth ResNet50-LSTM 权重。两个学习策略按官方发布说明使用 sampled
actions，种子为 1701/1702/1703。随机简单策略同样使用三种子；确定性简单策略
和 Oracle 只运行一个种子。

“有接触的 episode”是 episode 内至少一次 simulator collision 的比例，并非
逐步碰撞率。完整 collision count 保存在各方法原始 `summary.csv`。

完整性审计：每个有效方法/种子均有 495 行且 495 个唯一
`(scene_id, episode_id)`；所有可用种子的 episode 集与 ForwardOnly 的规范集
完全相同。有效官方闭环运行合计 7,425 个 episode。

## 3. 正式协议 B：MP3D 派生窄通道

数据来自正式 MP3D 扫描场景，但 episode 是本项目挖掘和生成的窄通道局部任务，
不能称为 Habitat 官方 PointNav benchmark。

- train：400 episode、44 个 MP3D train 场景；160 narrow、112 normal、
  48 wide、80 false-feasible 候选。
- val：80 episode、9 个 MP3D val 场景；32 narrow、22 normal、10 wide、
  16 false-feasible 候选。
- train/val 场景交集为 0。
- 机器人半径 0.18 m（直径 0.36 m），sliding 关闭，连续速度动作，最多 500 步。
- 80 个验证 episode 在 Geometry、kNN、DEGNAV-19D 和所有 E2E 种子之间固定。

### 3.1 19D 与记忆方法

| 方法 | N | All Success | Feasible Success | Correct Reject | Decision accuracy | Collision | Stuck | Avg steps |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometry-19D / no-memory | 80 | 82.50% | 98.44% | 0.00% | 78.75% | 0.00% | 1.25% | 136.03 |
| episodic kNN memory-19D | 80 | 82.50% | 98.44% | 0.00% | 78.75% | 0.00% | 1.25% | 136.03 |
| DEGNAV memory-19D | 80 | 82.50% | 98.44% | 0.00% | 78.75% | 0.00% | 1.25% | 136.03 |

三个方法完全相同不是“并列提升”，而是明确的负面结果。当前验证 episode 每次
只出现一次，kNN 和 DEGNAV 记忆写入量极少且未产生显式拒绝；因此这个协议不能
证明跨 episode 失败记忆有效。

### 3.2 DEGNAV-E2E 实现合同

部署 actor 为：`Raw normalized Depth (128×128) + PointGoal (2D) + previous
mode/collision/stuck/progress (7D) + GRU history → Commit/Explore/Recover/Reject`。
四模式再通过与教师相同的固定低层速度控制器执行。actor 不接收 19D、
false-feasible 标签或手工 4D task memory。

训练方式是离线 behavior cloning / policy distillation，不是 PPO。19D 和
false-feasible 标签只用于教师监督；视觉 backbone 从发布的 MP3D Depth
PointNav 权重严格加载 60 个 tensor。每个种子训练 20 epoch，batch size 2、
sequence length 16、stride 8、GRU hidden size 256；四模式按 inverse-frequency
power 1.0 等总损失权重训练。

修复后教师集的监督分布为：

| Split | Episodes | Frames | COMMIT | EXPLORE | RECOVER | REJECT | 历史错位 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| train | 400 | 28,162 | 18,640 | 6,550 | 2,898 | 74 | 0 |
| val | 80 | 5,206 | 4,347 | 714 | 132 | 13 | 0 |

教师验证参考达到 Feasible Success 100%、Feasible False Reject 0%、候选不可行
Correct Reject 81.25%。它使用特权 19D 和候选标签，只能作为训练上界，不能
作为可部署方法或公平 baseline。

| 方法 | Seeds | All Success | Feasible Success | Candidate traversal | Decision accuracy | Correct Reject | False Reject | Collision | Timeout | Steps |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEGNAV-E2E (Depth, offline BC) | 3 | 83.33±0.59% | 99.48±0.74% | 18.75±0.00% | 95.42±1.18% | 79.17±2.95% | 2.08±0.74% | 0.00±0.00% | 0.42±0.59% | 63.55±3.57 |

逐种子 All Success 为 `82.50/83.75/83.75%`，Feasible Success 为
`98.44/100.00/100.00%`，Correct Reject 为 `75.00/81.25/81.25%`。
终止结果合计为 200 次到达、39 次显式拒绝、1 次超时。所有种子都在候选不可行
子集上完成 3/16 次 traversal；因此 `false-feasible` 仍只能称生成器候选标签。

相对 Geometry-19D/no-memory，E2E 的 All Success 仅增加 0.83 个百分点，但
Decision accuracy 增加 16.67 个百分点、Correct Reject 增加 79.17 个百分点，
平均步数减少 72.48（53.28%）。这是“更会作决策和及时拒绝”的证据，不是与官方
PointNav 表之间的性能提升。代价是 2.08% False Reject。部署轨迹中 Commit、
Explore、Recover、Reject 分别出现于 220、50、1、42 个种子-episode；Recover
只在一个 episode 中出现且该 episode 最终超时，所以恢复能力尚未得到支持。

“E2E”只表示部署 actor 已去掉 19D，直接从原始 Depth、PointGoal 和动作结果历史
预测高层模式；连续速度仍由固定低层控制器实现，并不是像素直接回归速度的单体
端到端控制策略。

## 4. 发现并排除的问题

所有以下输出都保留在 `diagnostics/`，但不参与任何最终均值。

1. PointNav RGB-D channel order：旧 checkpoint 按 `RGB, Depth` 训练，当前 Gym
   plain dict 会排序为 `Depth, RGB`。修复采用 `OrderedDict`，并显式恢复 encoder
   的真实拼接列表；错误预检和 150/495 的部分运行已隔离。
2. Simple-agent 零视觉传感器崩溃：Habitat-Sim 0.3.3 在跨 MP3D 场景重配置时
   可能 segfault。策略仍不使用视觉，但 simulator 保留一个未使用的 Depth sensor。
3. E2E 教师无 EXPLORE：第一版 aligned anchors 导致 EXPLORE 支持为 0，未训练。
4. 不稳定 width proxy：第二版把零宽度伪影误当 EXPLORE，未训练。
5. 隐藏 passage-center REJECT：第三版依赖 actor 不可见的 passage-center 标注，
   中断并隔离。最终教师只在候选不可行且已经观察到 collision/stuck 后拒绝。
6. 弱 sqrt class balance：REJECT 的加权总损失仍只约为 COMMIT 的 5%，中断并
   替换为完整 inverse-frequency 权重。
7. 指标传感器误删：第一轮 E2E eval 为防 19D 泄漏而同时删除了
   `narrow_passage_success` 所需传感器，Success 被错误强制为 0。修复后传感器只
   服务环境指标，`actor_observation(include_privileged=False)` 保证不进入 actor。
8. 可变历史引用：采集器曾保存同一 `action_outcome` 数组的 view，后续原地更新
   把每个 episode 的全部历史覆盖成最终值，构成未来信息泄漏。该完整三种子结果
   已作废。现在每帧复制，并强制检查 frame 0 全零、frame t 的 previous-mode
   one-hot 等于 frame t−1 的教师动作；train/val 错位均为 0。

## 5. 可以与不能支持的论文结论

可以支持：

- 官方 PointNav PPO、DD-PPO、四种简单 agent 和 Oracle 已在同一 495-episode
  MP3D val 集完成，并有逐 episode 证据。
- 当前 MP3D 派生局部验证中，19D 几何控制可以高成功地通过可行候选。
- 原始深度 E2E actor 的输入边界、训练标签来源和指标侧特权传感器已机器审计。
- 原始深度 E2E 在三个种子上保持 99.48±0.74% Feasible Success，并把显式
  Correct Reject 从 19D 三行的 0% 提高到 79.17±2.95%。

不能支持：

- 不能把官方 PointNav 数值与派生 DEGNAV 数值直接比较或计算提升百分点。
- 不能声称 kNN/DEGNAV memory 优于 no-memory；当前三行完全相同。
- 不能把 `false-feasible` 候选标签当成接触仿真证明的绝对不可行真值。
- 不能把 traversal failure、timeout 或 stuck 自动记为 Correct Reject；只有显式
  Reject 才计数。
- 当前 E2E 是 BC，不得写成 end-to-end PPO 或 online RL。
- 当前 E2E checkpoint 使用 MP3D val 教师分类指标选 epoch，并在同一 val episode
  做闭环评测；它是 scene-disjoint validation，不是独立 test-set 结论。
- 不能声称四个模式都已经可靠工作：Recover 仅在 240 个种子-episode 中出现一次，
  且该次最终超时。

## 6. 后续改进优先级

1. 对四模式 actor 做 DAgger：在 actor 自己访问到的状态上由教师重新标注，解决
   teacher-forcing 和闭环分布偏移。
2. 以 BC checkpoint 初始化四模式 PPO，仅学习高层模式，低层控制器固定；reward
   同时覆盖 progress、alignment、body margin、恢复成功、误拒和正确拒绝。
3. 对模式切换首帧进行 transition-aware sampling/loss。当前大量 RECOVER 帧是
   已经进入恢复后的持续帧，真正的“启动 Recover”监督远少于逐帧计数。
4. 增加真实失败证据覆盖：扩展不可行和临界可行入口，成对采集相同几何下的
   Recover/Reject，避免网络只学类别频率。
5. 加入辅助几何蒸馏头（clearance/body margin/heading），训练时使用 19D 监督，
   部署时仍只输入原始 Depth；与纯 BC 做消融。
6. 增加单独的 STOP head 或经过审计的停止条件，避免把“到达目标”和四种风险
   模式混为一个输出。
7. 建立 train/selection/test 三段场景划分，再把最终模型放到未参与 checkpoint
   选择的场景上评测。
8. 记忆贡献需要 repeated-passage 协议：同一候选多次访问、相似候选迁移和无关
   候选干扰必须同时测量。

## 7. 复现入口

官方 PointNav 全序列（学习策略优先）：

```bash
cd /home/xiaotian/navigation/habitat-lab
bash examples/narrow_passage_rl/run_official_pointnav_sequence.sh
```

派生 19D、E2E 教师、三种子训练和评测：

```bash
cd /home/xiaotian/navigation/habitat-lab
bash examples/narrow_passage_rl/run_degnav_e2e_sequence.sh
```

两个脚本均有 episode 数、种子数、同集和部署输入合同检查；E2E 脚本还检查
逐帧历史对齐并可从完整阶段断点续跑。

## 8. 证据文件

- 官方总表：`../official_pointnav_mp3d_v1_20260904/paper_table_official_pointnav.md`
- 官方 manifest：`../official_pointnav_mp3d_v1_20260904/summary_manifest.json`
- 派生 19D 表：`../degnav_e2e_mp3d_narrow_v1_20260904/19d_comparison/formal_mp3d_comparison.md`
- 派生 E2E 表：`../degnav_e2e_mp3d_narrow_v1_20260904/paper_table_degnav_e2e.md`
- E2E 三种子汇总：`../degnav_e2e_mp3d_narrow_v1_20260904/summary_by_seed.csv`
- E2E 审计、模式覆盖和哈希：`../degnav_e2e_mp3d_narrow_v1_20260904/summary.json`
- 最终方法状态：`final_method_status.md`
- 环境与 SHA-256：`environment_snapshot.md`

环境详情见 `environment_snapshot.md`。MP3D 场景和发布 checkpoint 是外部大文件，
不应提交 Git；代码、表格、逐 episode CSV、汇总 JSON、日志和无效运行说明可以
提交。当前 E2E 结果目录约 615 MB，其中每个 `best.pt`/`last.pt` 约 40 MB，并含
无效诊断 checkpoint；普通 Git 提交前应只选择最终 checkpoint，或使用 Git LFS /
Release 存储。是否已经推送远端必须由 `git status` 和远端 commit 单独确认，不能
仅凭本地文件存在来声称“已上传 GitHub”。
