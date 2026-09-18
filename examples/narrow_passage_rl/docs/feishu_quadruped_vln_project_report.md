# 面向复杂环境的四足机器人视觉语言导航与强化学习自主运动系统

> 文档用途：导师汇报、项目阶段评审、组会讨论和论文项目管理
> 文档风格：ICRA / IROS / AAAI 机器人导航项目报告
> 校对日期：2026-07-24
> 数据原则：只引用仓库内可追溯结果；未完成实验统一标记为“进行中”或“未来计划”

## 文档状态与证据边界

| 内容 | 状态 | 说明 |
|---|---|---|
| Procedural v2 窄通道评测 | 已完成 | 具有 episode-level CSV、3 seeds 和可复现表格 |
| Habitat-Lab HM3D nominal / stress 评测 | 已完成 | nominal anchor、yaw、横向偏移和 extreme-narrow 等结果已记录 |
| Failure-aware memory recurrence / transfer | 已完成 | 仅支持合成 memory protocol 下的结论 |
| DEGNAV-RL 模式选择策略 | 诊断性结果 | 当前未学习出可靠 Recover / Reject，不作为主方法 |
| Vision-Language Navigation | 进行中 | 有 NaVILA runtime、640-case diagnostic 和 adapter smoke；尚无标准 VLN SR、SPL、NE、nDTW 结果 |
| 四足机器人 3D 导航 | 未来计划 | 已形成 Point Cloud、3D Occupancy 与 Geometry Token 技术方案，尚无正式实验结果 |
| Isaac Sim 四足运动控制 | 用户报告已完成，待归档 | 尚无可核验的训练曲线、checkpoint 和多 seed 结果 |
| DeepRobotics Lite3 闭环实验 | 用户报告已完成，待归档 | 尚缺 rosbag、重复统计和完整结果表 |

本文档中的 Habitat clearance、near-collision 和 strict success 来自深度观测与近似机身余量模型，只作为 clearance-aware diagnostic，不解释为经过标定的真实物理安全测量。

## 飞书目录

1. 项目首页
2. 研究背景与问题定义
3. 系统技术路线
4. Vision-Language Navigation
5. Habitat-Lab 与 Geometry-aware Navigation
6. Failure-aware Memory
7. 强化学习与 Isaac Sim 四足运动控制
8. DeepRobotics Lite3 实机验证
9. Sim-to-Real 与系统集成
10. 实验结果分级与论文叙事
11. 项目风险与解决计划
12. 时间计划模板
13. 飞书素材上传清单
14. 仓库结果与文档索引
15. 汇报页排版建议
16. Future Work：面向四足机器人的 3D 导航
17. 机器人项目总结

# 第一部分 项目首页

## 1.1 项目简介

本项目面向未知室内环境、狭窄通道、家具遮挡和动态干扰条件下的四足机器人自主导航问题，研究视觉语言理解、显式空间几何推理、失败记忆与强化学习运动控制的协同方法。系统通过 Vision-Language Navigation 将自然语言指令转化为语义目标与局部航点，利用几何可行性估计和失败记忆判断通道风险，通过 Commit、Explore、Recover、Reject 四类模式生成局部行为，并由 Isaac Sim 中训练的运动控制策略和 ROS2 控制接口驱动 DeepRobotics Lite3 执行。当前重点是建立可解释、可复现的 Habitat-Lab 窄通道评测体系，并逐步完成仿真到实机迁移。

## 1.2 核心研究问题

1. 如何将自然语言目标转化为四足机器人可执行的语义目标、子目标和局部航点？
2. 如何在接近机器人几何可行性边界时估计通道可通过性及其不确定性？
3. 如何利用跨 episode 失败记忆减少对历史不可行通道的重复尝试？
4. 如何协调高层导航决策与四足机器人低层强化学习运动技能？
5. 如何处理 Habitat、Isaac Sim 与 Lite3 之间的观测、动力学和控制接口差异？

## 1.3 技术路线总结

```text
自然语言指令 + RGB-D + 位姿
              |
              v
Vision-Language Navigation
语义目标 / 子目标 / 局部航点
              |
              v
Geometry Encoder + Feasibility Belief + Failure Memory
              |
              v
Risk-aware Mode Decision
Commit / Explore / Recover / Reject
              |
              v
Mode-conditioned Local Controller
速度命令 / 局部轨迹 / 运动技能
              |
              v
RL Locomotion + IMU + Joint State + Contact
              |
              v
DeepRobotics Lite3 四足运动
              |
              v
碰撞 / 卡滞 / 恢复 / 成功结果写入 Failure Memory
```

系统由三层组成：语义目标层负责理解“去哪里”；几何决策层负责判断“能否进入、是否需要探索、恢复或拒绝”；运动控制层负责实现“如何稳定移动”。

## 1.4 当前阶段成果

| 状态 | 工作内容 | 当前证据 |
|---|---|---|
| 已完成 | Habitat-Lab 自定义窄通道任务 | 支持 HM3D anchor、RGB-D 视频、stress validation 和 episode 日志 |
| 已完成 | Procedural v2 benchmark | 支持直道、L/S 弯道、窄入口、窄出口、非对称和 false-feasible |
| 已完成 | DEGNAV-Rule / Geometry-FSM | Overall SR 为 `70.3±1.4%`，Reactive rule 为 `25.4±1.2%` |
| 已完成 | 核心消融 | w/o alignment 为 `2.4±0.5%`；w/o recovery 与 full 基本相同 |
| 已完成 | False-feasible outcome decomposition | DEGNAV-Rule 降低 hard collision，但 explicit / correct reject 仍为 `0%` |
| 已完成 | Memory transfer / interference protocol | 验证重复失败抑制、相似不可行迁移与相似可行干扰 |
| 诊断性结果 | DEGNAV-RL | 当前模式塌缩到 Commit / Explore，不作为竞争性最终方法 |
| 进行中 | VLN 接口 | 需要补充标准 VLN 指标及未见场景结果 |
| 待归档 | Isaac Sim locomotion | 用户报告训练已完成；需要补充训练预算、多 seed、扰动和域随机化结果 |
| 待归档 | Lite3 实机闭环 | 用户报告验证已完成；需要补充重复实验、rosbag、失败案例和完整视频 |

## 1.5 当前实验平台

| 平台 | 作用 | 当前状态 |
|---|---|---|
| Habitat-Lab / Habitat-Sim / HM3D | 场景级导航、窄通道 anchor 和扰动验证 | 已完成主要评测流程 |
| Procedural v2 | 可控生成几何边界、弯道和 false-feasible 场景 | 已完成主要评测 |
| Stable-Baselines3 | PPO、SAC、TD3 direct-control baseline 和 DEGNAV-RL | 已完成部分正式与诊断实验 |
| Isaac Sim | 四足动力学、运动技能、域随机化与扰动测试 | 用户报告训练已完成，证据待归档 |
| ROS2 | 感知、决策、控制和数据记录接口 | 进行中 |
| DeepRobotics Lite3 | 真实四足机器人闭环执行 | 用户报告验证已完成，证据待归档 |

## 1.6 下一步研究计划

1. 完成 VLN 输出到局部航点和几何决策层的统一接口。
2. 为 Recover / Reject 增加专用监督、奖励和困难样本，解决模式塌缩。
3. 完成 Isaac Sim locomotion curriculum、域随机化和多 seed 训练。
4. 建立 Habitat、Isaac Sim 和 Lite3 一致的动作、坐标系与约束定义。
5. 在 Lite3 上完成窄通道、错误通道、恢复和语言目标导航的重复实验。
6. 统一论文表格、视频、rosbag、checkpoint 和复现命令。
7. 建立 RGB-D 点云、3D occupancy 与 Geometry Token 管线，评估其相对 2D costmap 的增益。

# 第二部分 研究背景与问题定义

## 2.1 研究背景

复杂环境中的自主导航是移动机器人研究的核心问题之一。传统系统通常由 SLAM、全局路径规划与局部运动控制构成，例如 ROS2 Nav2 集成的 DWA、MPPI，以及基于模型预测控制的 MPC 方法。这类方法具有模块清晰、行为可解释和工程部署成熟等优势，在地图可靠、障碍物结构规则且机器人运动模型相对稳定的场景中表现良好。然而，其决策主要依赖几何地图、局部代价函数和人工设定规则，通常缺少对自然语言指令、物体类别、空间功能和任务语义的联合理解。

当机器人进入未知、半结构化或高度受限的室内环境时，传统导航管线的局限性进一步显现。在线建图误差、深度噪声、局部遮挡和动态障碍会降低代价地图与局部规划器的可靠性；在狭窄入口、非对称障碍和弯曲通道附近，局部目标与避障代价可能相互竞争，使机器人产生左右振荡、频繁重规划、局部死锁或反复进入不可行路径。传统控制器通常还将每次失败视为独立事件，难以利用先前碰撞、卡滞和恢复经历调整后续决策。

Vision-Language Navigation 为复杂任务提供了语义层面的导航能力，使机器人能够根据自然语言指令、视觉观测和环境上下文推断目标对象、空间关系与局部子目标。Memory-based Decision 则允许系统保留历史失败案例，并根据当前几何状态与历史经验的相似性估计重复失败风险。二者结合后，机器人不仅需要理解“去哪里”，还需要判断“当前通道是否值得进入”以及“是否应探索、恢复或拒绝”。

对于四足机器人，上述问题还需要与低层运动动力学共同考虑。与主要在平面运动的轮式机器人相比，四足机器人具有更复杂的质心运动、姿态稳定、足端接触和关节力矩约束。导航层给出的速度命令并不必然对应稳定可执行的运动，狭窄空间中的转向、侧向摆动和接触变化还可能导致机身碰撞或步态失稳。因此，需要通过 Reinforcement Learning Locomotion 学习具备扰动恢复和接触适应能力的运动技能，并通过域随机化、传感器噪声建模和控制时延建模降低仿真到真实机器人迁移时的性能退化。

## 2.2 研究目标

本项目不以构建单一端到端策略为目标，而是研究语言理解、几何可行性、失败历史和四足动力学如何在一个可解释系统中协同工作。核心目标包括：

1. 建立从自然语言指令到语义目标和局部航点的 VLN 接口。
2. 建立机器人形态约束下的通道可行性 belief 与风险估计。
3. 建立跨 episode failure memory，减少重复不可行 commitment。
4. 建立高层模式决策与低层 RL locomotion 的分层接口。
5. 建立 Habitat、Isaac Sim 和 Lite3 的可复现实验链路。

## 2.3 关键挑战

| 挑战 | 具体问题 | 计划解决方式 |
|---|---|---|
| 语义与几何脱节 | VLN 航点语义正确但物理不可通行 | Geometry-aware waypoint validation |
| 几何边界不确定 | 深度噪声与入口角度改变有效机身宽度 | Feasibility belief 与 yaw-aware width |
| 重复失败 | 控制器无法利用历史失败 | Geometry-guided cross-episode memory |
| 模式语义难学习 | RL 容易只使用 Commit / Explore | 模式监督、奖励分解和困难样本 |
| 四足执行约束 | 速度命令不等于稳定步态 | RL locomotion 与接触反馈 |
| Sim-to-Real | 视觉、动力学、时延和控制频率不同 | 域随机化、接口统一和分阶段验证 |

# 第三部分 系统技术路线

## 3.1 系统架构描述

本文采用由语义理解、风险决策和机器人控制构成的三级分层架构。高层 Vision-Language Navigation 将语言指令和视觉观测转换为语义目标与局部航点；中层 Risk-aware Navigation 根据通道几何、机器人形态和失败记忆选择离散决策模式；低层 Robot Control 将导航指令转换为满足四足机器人动力学与接触约束的运动行为。

系统映射可写为：

```text
(Language, RGB, Depth, Pose)
        -> (Semantic Goal, Waypoint)
        -> Mode
        -> Velocity / Skill Command
        -> Quadruped Motion
```

其中，模式属于 `{Commit, Explore, Recover, Reject}`。

## 3.2 Layer 1：Vision-Language Navigation

### 研究目标

根据自然语言指令和 RGB-D 观测生成语义目标、局部子目标与航点。

### 输入与输出

| 类型 | 内容 |
|---|---|
| 输入 | Language Instruction、RGB、Depth、Robot Pose |
| 输出 | Semantic Goal、Waypoint、Subgoal |

### 模块作用

1. 提取语言中的目标实体、空间关系和任务约束。
2. 建立语言与视觉区域之间的跨模态对应。
3. 预测目标位置或候选局部航点。
4. 将长时语言任务分解为可执行子目标。

## 3.3 Layer 2：Risk-aware Navigation

### 研究目标

将语义航点转换为满足机器人尺寸、入口姿态和局部风险约束的决策模式。

### 输入与输出

| 类型 | 内容 |
|---|---|
| 输入 | Geometry Feature、Passage Width、Robot Shape、Failure Memory |
| 输出 | Commit、Explore、Recover、Reject |

### 模式定义

| 模式 | 触发条件 | 行为 |
|---|---|---|
| Commit | 可行性高且风险低 | 稳定执行通过 |
| Explore | 信息不足但风险可控 | 低速试探并获取新观测 |
| Recover | 检测到卡滞、振荡或局部失败 | 后退、重对齐或重新定位 |
| Reject | 几何不可行或历史失败风险高 | 停止进入并请求重新规划 |

### Feasibility Belief

当前 compact belief state 表示为：

```text
b_t = (
  p_feas,
  delta_mean,
  delta_var,
  heading_error,
  lateral_error,
  stuck_score,
  collision_flag,
  memory_risk
)
```

其中，`delta_mean = d_hat - w_req_cons` 表示估计通道宽度与保守所需宽度之间的余量。`p_feas` 反映在宽度估计和机器人形态不确定性下的可行概率。

## 3.4 Layer 3：Robot Control

### 研究目标

根据高层模式与本体状态生成稳定、可执行的四足运动。

### 输入与输出

| 类型 | 内容 |
|---|---|
| 输入 | Velocity Command、IMU、Joint State、Contact State |
| 输出 | Joint Target、Foot Trajectory、Torque 或 Quadruped Motion |

### 模块作用

1. 跟踪线速度和角速度命令。
2. 生成步态并维持机身姿态稳定。
3. 根据足端接触和打滑状态调整运动。
4. 在外部扰动、碰撞或卡滞后执行恢复。
5. 将导航模式映射为不同速度范围和运动技能。

## 3.5 模块间数据流

1. VLN 根据语言与 RGB-D 观测生成语义目标和局部航点。
2. 几何模块围绕航点提取通道宽度、左右 clearance、入口角度和机身余量。
3. 风险决策模块融合几何 belief 与失败记忆，选择高层模式。
4. 模式条件控制器将模式转换为速度命令或运动技能。
5. 四足控制器根据 IMU、关节状态和接触信息执行运动。
6. 碰撞、卡滞、振荡、恢复和拒绝结果写入 Failure Memory。
7. 位姿和执行偏差反馈给 VLN 与局部导航层。

## 3.6 为什么采用分层结构

语言理解、几何决策和四足控制具有不同时间尺度和约束形式。语言理解以任务或子目标为单位更新；几何风险随局部观测更新；四足控制则必须以高频率处理姿态与接触。分层结构能够隔离这些时间尺度，使高层目标不承担关节级控制学习负担，也使低层策略不必直接从原始语言推断运动。

分层设计还便于分别验证语义理解、几何误判和低层执行失败，并允许替换 VLN 模型、风险决策器或 locomotion policy，而无需重构整个系统。

## 3.7 相比端到端方法的优势

1. 显式加入机器人尺寸、通道宽度和姿态约束。
2. 可以解释 Commit、Explore、Recover 和 Reject 的触发原因。
3. Failure Memory 能够独立更新，无需重新训练完整策略。
4. Habitat 中的导航决策与 Isaac Sim 中的运动控制可以分别训练。
5. 更容易定位语义、几何、动力学和 sim-to-real 误差来源。
6. 便于实施速度限制、安全停止和人工接管等工程约束。

上述优势是系统设计动机，不等价于已通过实验验证分层方法在所有场景中都优于端到端方法。后续仍需设置统一数据与预算下的对照实验。

# 第四部分 Vision-Language Navigation

## 4.1 Background

Vision-Language Navigation 研究智能体如何依据自然语言指令和视觉观测在未知环境中完成目标驱动导航。现有方法通常通过视觉编码器、语言编码器和跨模态注意力学习指令与轨迹之间的对应关系，并直接预测离散导航动作或局部航点。这类方法在标准室内数据集上建立了较完整的评测体系，但训练目标通常以轨迹模仿或终点到达为主，对三维可通行空间、机器人真实形态和低层运动约束考虑不足。

对于四足机器人，仅预测语义正确的目标或视觉上合理的轨迹并不足以保证任务完成。候选路径可能穿过小于机身包络的通道，局部航点也可能要求机器人以不可稳定执行的入口角度转向。此外，视觉域差异、深度噪声、相机安装位置和实际控制时延都会加剧从 VLN 仿真评测到真实机器人部署的性能下降。

## 4.2 Problem Definition

给定自然语言指令 `L`，机器人在时刻 `t` 获得观测：

```text
o_t = {I_rgb_t, D_t, p_t, s_t}
```

其中，`I_rgb_t` 为 RGB 图像，`D_t` 为深度观测，`p_t` 为机器人位姿，`s_t` 为机器人状态。VLN 策略根据当前观测与历史信息输出导航动作、局部航点或语义子目标：

```text
(a_t, w_t, g_t) = pi_VLN(L, o_<=t)
```

对于四足机器人，策略不仅需要最小化目标距离，还应考虑路径可通行性、机身包络、入口姿态、碰撞风险和低层控制可执行性。

### 数据和指标

| 指标 | 含义 | 当前状态 |
|---|---|---|
| Success Rate | 是否到达语言指定目标 | 待评测 |
| SPL | 成功率与路径效率 | 待评测 |
| Navigation Error | 终点与目标距离 | 待评测 |
| Oracle Success | 轨迹是否曾到达目标邻域 | 待评测 |
| Waypoint Error | 预测航点与参考航点偏差 | 待定义 |
| Inference Latency | VLN 前向推理时间 | 已测：NaVILA 4-bit 平均 `1.466 s`，P95 `1.485 s` |
| Action Parse Rate | 输出是否可解析为局部动作 | 已测：640-case diagnostic 为 `100.0%` |
| Stop Override Compliance | 显式停止覆盖是否生效 | NaVILA direct `1.6%`，adapter smoke `100.0%` |
| Visual Perturbation Consistency | 扰动前后动作一致性 | 已测：非 clean perturbation `75.0%` |

## 4.3 Proposed Framework

本文采用 Hierarchical VLN：

```text
Language Understanding
        |
        v
Semantic Goal Generation
        |
        v
Geometry-aware Navigation
        |
        v
RL Controller
```

Language Understanding 提取目标实体、空间关系和任务约束；Semantic Goal Generation 将语言语义与视觉观测对齐，生成目标位置和局部航点；Geometry-aware Navigation 根据深度、通道宽度和机器人形态判断航点的物理可执行性；RL Controller 根据速度命令、本体感知和接触状态生成四足运动。

### 实验设计

1. 比较 oracle goal、PointNav goal 和 VLN-predicted goal。
2. 分别报告 seen 与 unseen scene 结果。
3. 将错误拆分为语言理解、语义定位、局部几何决策和运动执行。
4. 在相同航点上比较 geometry-aware 与不含几何约束的局部控制。
5. 测试语言歧义、目标遮挡和目标不可达场景。

### 图片建议

- 自然语言指令、RGB-D 图像和预测目标的组合图。
- Top-down map 上的语义目标、候选航点与执行轨迹。
- 成功案例与语言理解失败案例的时间序列。
- VLN 输出与 Geometry-aware Navigation 接口图。

### 当前结果分析

当前仓库尚无标准 VLN SR、SPL、Navigation Error、nDTW 或 unseen-scene
paired simulator 数据，因此不能把当前结果解释为正式 VLN benchmark。
但已完成 NaVILA 本地模型加载、640-case controlled diagnostic 和
NaVILA-to-DEGNAV safety adapter smoke test。结果显示：模型输出动作可解析率
`100.0%`，但显式 stop override compliance 仅 `1.6%`，R2R text probe 中
move-forward 输出占 `71.5%`，controlled left/right/stop match 为 `0.0%`，
非 clean 视觉扰动动作一致率为 `75.0%`。加入确定性安全适配器后，在相同
640 条保存输出上，受控动作匹配率由 `25.0%` 提升到 `81.3%`，trusted stop
compliance 由 `1.5%` 到 `100.0%`，四个 synthetic blocked-forward probe 的
forward 输出由 `100.0%` 降到 `0.0%`，同时保留所有 256 条原始未受信任 R2R
route proposal。该结果属于接口级 smoke evidence，不是 learned VLN performance。

## 4.4 Future Direction

1. 使用三维场景图或开放词汇地图增强语言目标与空间结构关联。
2. 将 VLN 航点置信度与几何可行性 belief 联合建模。
3. 在局部几何不可观测时引入基于信息增益的主动探索。
4. 建立语言指令、失败记忆和环境变化之间的长期关联。
5. 通过视觉域随机化与真实数据适配降低 sim-to-real 差异。

# 第五部分 Habitat-Lab 与 Geometry-aware Navigation

## 5.1 研究目标

在 procedural v2 和 HM3D 场景中验证接近几何可行性边界的四足机器人窄通道导航，重点分析入口对齐、弯道跟踪、非对称障碍、false-feasible 与扰动敏感性。

## 5.2 方法介绍

### Geometry-FSM 与 DEGNAV-Rule

`Geometry-FSM` 是代码实现名，论文中称为 `DEGNAV-Rule`。该方法使用显式几何观测、yaw-aware required width、feasibility belief 和模式条件控制器。

### Yaw-aware Required Width

机器人在入口偏航角 `theta` 下的平面机身包络近似为：

```text
w_body(theta) = A |cos(theta)| + B |sin(theta)|
```

传感误差、滚转和执行余量作为加性 margin 作用于 required width，不改变其随 yaw 变化的基本形状。

### DEGNAV-RL

DEGNAV-RL 学习高层模式策略 `pi(m_t | b_t)`，不直接输出速度。当前结果显示该策略主要使用 Commit 和 Explore，没有学习出可靠 Recover / Reject，因此只作为诊断性策略，不是主方法。

### Direct-control RL Baselines

PPO、SAC 和 TD3 direct-control baseline 将几何观测直接映射为线速度和角速度。它们用于分析 learning-only 策略在几何边界与 synthetic-to-Habitat transfer 中的表现，不作为本文主贡献。

## 5.3 实验设计

### Procedural v2

包含以下七类通道：

| 类型 | 目的 |
|---|---|
| Straight | 基础窄通道通过 |
| L-shaped | 单次转弯与局部目标跟踪 |
| S-shaped | 连续多转弯 |
| Narrow entry | 入口对齐 |
| Narrow exit | 出口 clearance |
| Asymmetric | 左右 clearance 不均衡 |
| False-feasible | 入口看似可行但内部不可行 |

### Habitat HM3D

Habitat 实验拆分为：

1. Nominal same-split anchor validation。
2. Yaw、横向偏移、极窄子集、depth dropout 和 feature noise stress。
3. Clearance-aware diagnostic。

## 5.4 数据和指标

| 指标 | 定义 | 解释限制 |
|---|---|---|
| Success | 满足任务距离与对齐条件 | 主指标 |
| Collision | 仿真器或任务碰撞标志 | 与边界停止机制相关 |
| Timeout / Stuck | 超时或低进度卡滞 | 与 Reject 分开 |
| Near collision | body-margin proxy 低于阈值 | 仅 diagnostic |
| Strict success | Success 且无碰撞/卡滞并满足 clearance | 仅 diagnostic |
| Minimum clearance | episode 内最小 body-margin proxy | 可能受深度伪影影响 |

## 5.5 Procedural v2 主结果

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible traversal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reactive rule baseline | 25.4±1.2% | 60.7±3.8% | 8.4±3.1% | 6.4±2.7% | 27.8±5.9% | 24.8±1.6% | 33.6±4.6% | 0.0±0.0% |
| DEGNAV-Rule | 70.3±1.4% | 92.8±2.6% | 79.4±7.0% | 70.3±4.4% | 96.5±2.0% | 61.7±9.3% | 50.3±3.9% | 0.0±0.0% |

实验预算为每个方法每个 seed 500 episodes，seeds 为 42、43、44。False-feasible traversal success 为 0% 不能解释为 correct rejection。

## 5.6 核心消融

| Variant | Overall | Delta vs full | Collision | Timeout/stuck |
|---|---:|---:|---:|---:|
| DEGNAV full | 69.9±1.6% | +0.0±0.0 pp | 17.0±1.4% | 13.1±1.9% |
| w/o alignment | 2.4±0.5% | -67.5±2.1 pp | 48.1±1.8% | 49.5±1.5% |
| w/o recovery | 69.9±1.6% | +0.0±0.0 pp | 17.0±1.4% | 13.1±1.9% |
| deterministic margin | 69.3±1.5% | -0.6±0.2 pp | 17.8±1.6% | 12.9±2.1% |
| w/o yaw prior | 70.1±1.7% | +0.1±0.4 pp | 16.9±1.0% | 13.1±1.9% |

当前 benchmark 识别出 alignment 是主要 measured component。Deterministic-margin 和 no-yaw-prior 与 full 接近，因此当前实验没有充分隔离 probabilistic belief 与 yaw prior 的贡献。w/o recovery 与 full 相同，说明该 nominal procedural protocol 没有充分激活 recovery，不能据此否定恢复模块。

## 5.7 Habitat same-split nominal anchor validation

| Method | Success | Steps | Episodes |
|---|---:|---:|---:|
| PPO direct-control baseline | 10.2% | 498.0 | 157 |
| APF+Gap | 93.6% | 111.5 | 157 |
| DEGNAV-Rule | 100.0% | 54.0 | 157 |
| DEGNAV-Rule + failure memory | 100.0% | 54.0 | 157 |

上述数据来自 HM3D Val set A 的相同 157 个 episode IDs。DEGNAV-Rule 的 100% 必须描述为“当前 mining protocol 下的 nominal anchor validation”，不能描述为完整鲁棒性、完美泛化或解决 HM3D 窄通道问题。Memory 在 one-shot passable anchor 上没有显示增益。

## 5.8 Habitat stress key slices

| Stress | Method | Episodes | Success | Timeout | Avg steps |
|---|---|---:|---:|---:|---:|
| nominal | APF+Gap | 151 | 96.7% | 0.0% | 94.8 |
| nominal | DEGNAV full | 151 | 100.0% | 0.0% | 48.1 |
| yaw60 | DEGNAV full | 151 | 100.0% | 0.0% | 51.5 |
| yaw60 | w/o heading alignment | 151 | 76.8% | 16.6% | 160.1 |
| extreme_yaw60 | DEGNAV full | 24 | 100.0% | 0.0% | 50.3 |
| extreme_yaw60 | w/o heading alignment | 24 | 79.2% | 12.5% | 138.3 |
| lat020 | DEGNAV full | 151 | 98.0% | 0.0% | 47.9 |

该表支持“controlled perturbation 下的 module sensitivity”，不支持普遍鲁棒性声明。

## 5.9 图片建议

- Procedural 七类通道俯视图。
- HM3D RGB、depth 与 top-down trajectory。
- DEGNAV-Rule 在 yaw60 下的进入、对齐、Commit 和完成关键帧。
- Yaw-aware required-width 曲线。
- Rule vs DEGNAV margin-phase 图。
- Nominal、stress 和 clearance diagnostic 三类表格关系图。

## 5.10 结果分析

现有结果支持：显式几何对齐和模式控制在当前窄通道 benchmark 中优于 reactive rule；heading alignment 在 yaw stress 下具有明显作用。现有结果不支持：DEGNAV-Rule 已解决 false-feasible rejection、Habitat 100% 等价于普遍鲁棒性、clearance proxy 等价于真实物理安全。

## 5.11 Habitat-Lab Experiments（ICRA / IROS 章节格式）

### 5.11.1 Simulation Environment

实验平台基于 Habitat-Lab 与 Habitat-Sim，室内场景采用 HM3D。Habitat-Lab 提供任务、数据集、传感器、动作和度量接口，Habitat-Sim 负责加载三维扫描场景、执行机器人运动并返回视觉与导航状态。当前仓库在标准导航接口之外新增 `NarrowPassageNav-v0`，用于研究机器人接近几何可行性边界时的入口对齐、通道穿越、卡滞恢复与错误通道处理。

本项目在 Habitat 中区分三类实验，避免将不同协议下的数据混排：

| 实验类别 | 目的 | 当前状态 |
|---|---|---|
| PointNav | 验证已知相对目标下的基础导航能力 | 已完成 Habitat test-scenes 50-episode 单 checkpoint pipeline evaluation；正式 HM3D multi-seed 评测待补 |
| ObjectNav | 验证目标类别驱动的语义导航能力 | 已完成 HM3D ObjectNav v2 `val_mini` 20-episode single-run smoke evaluation；held-out generalization 待补 |
| Narrow Passage Navigation | 验证几何边界、模式决策和扰动敏感性 | 已完成 procedural v2、HM3D nominal 与 stress 评测 |

HM3D nominal comparison 使用相同 split 和相同 episode IDs。当前 canonical nominal comparison 为 HM3D Val set A 的 157 个 episodes。Stress validation 使用独立的受控扰动协议，不与 nominal table 直接混排。Clearance-aware strict success、near collision 和 minimum clearance 单独作为诊断指标。

### 5.11.2 Task Definition

#### PointNav

给定机器人初始状态和相对目标位置，策略根据视觉、深度和定位观测生成导航动作，使机器人到达目标邻域。任务可表示为：

```text
pi(a_t | RGB_t, Depth_t, GPS_t, Compass_t, goal)
```

PointNav 主要用于分离语义理解误差与基础导航误差。当前保留的 TensorBoard aggregate 和 50 个 episode 视频相互核验得到：Success `94.0%`、SPL `0.792`、最终 Distance-to-Goal `0.114 m`。该结果来自 `ppo_pointnav_example.yaml`、Habitat test-scenes val 和单个约 0.98M-frame checkpoint，只能说明标准 PointNav pipeline 已跑通；由于缺少多 seed HM3D held-out 评测和正式 per-episode CSV，不能作为主论文的通用导航性能结论。

#### ObjectNav

给定目标物体类别，机器人需要在未知室内场景中探索、定位目标实例并导航至可观察位置。策略可表示为：

```text
pi(a_t | RGB_t, Depth_t, GPS_t, Compass_t, object_category)
```

ObjectNav 用于验证语义目标定位能力，并为后续 VLN 的 language-to-object grounding 提供中间基线。当前保留的单次 TensorBoard evaluation aggregate 对应 20 个 HM3D ObjectNav v2 `val_mini` episodes：Success `15.0%`、SPL `0.113`、SoftSPL `0.640`、最终 Distance-to-Goal `1.074 m`，checkpoint 约为 17.1M frames。该实验是 local smoke run，不能证明 held-out scene generalization；正式实验仍需固定 train/val split、保存 per-episode CSV 并运行多 seed。

#### Narrow Passage Navigation

给定局部目标与场景几何，机器人需要在满足距离、朝向和横向误差约束的同时完成通道穿越。当前 Habitat task 为 `NarrowPassageNav-v0`，成功判定参数为：

| 条件 | 当前配置 |
|---|---:|
| Goal distance | `<= 0.50 m` |
| Heading error | `<= 0.52 rad` |
| Lateral error | `<= 0.40 m` |
| Stop action | 不要求 |

该任务不仅考察是否到达目标，还记录 passage width、body margin、heading error、lateral offset、stuck score 和 collision flag。False-feasible 场景必须进一步区分 explicit reject、collision、timeout 和 stuck；`success == false` 不等价于正确拒绝。

### 5.11.3 Observation Space

面向完整导航系统，统一观测接口计划包含 RGB、Depth、GPS、Compass 与 Robot State：

| 观测 | 作用 | 当前使用情况 |
|---|---|---|
| RGB | 物体、材质、语义目标和视觉上下文 | 视频与 VLN 接口使用；VLN 正式结果待补 |
| Depth | 局部自由空间、障碍距离与通道几何 | Narrow Passage 几何特征的主要来源 |
| GPS | 相对位移或全局位置估计 | PointNav / ObjectNav 标准接口 |
| Compass | 机器人朝向和目标方位 | PointNav / ObjectNav 标准接口 |
| PointGoal | 局部目标距离与角度 | Narrow Passage 当前任务使用 polar 2-D goal |
| Geometry feature | 19-D 可解释局部几何状态 | DEGNAV 与 direct-control RL baseline 使用 |
| Memory feature | 4-D failure-memory 状态 | Memory evaluator 使用；nominal smoke 默认可关闭 |

当前 19-D `narrow_passage_features` 布局为：

```text
[d_left_near, d_center_near, d_right_near,
 d_left_far, d_center_far, d_right_far,
 clearance_left, clearance_right,
 passage_width, body_margin,
 heading_error, lateral_offset, distance_to_local_goal,
 current_vx, current_wz, stuck_score, collision_flag,
 previous_action_vx, previous_action_wz]
```

4-D `narrow_passage_memory` 包含：

```text
[failure_count, failed_state_count, should_recover, should_reject]
```

需要特别说明，当前 Narrow Passage direct-control policy 的主要输入是由 depth、goal 和本体状态提取的 19-D 几何特征，而不是将原始 RGB-D 直接输入网络。因此其结果应称为 geometry-sensor local-control baseline，不能表述成完整视觉 PointNav 或 ObjectNav 策略。

### 5.11.4 Action Space

PointNav 与 ObjectNav 可采用 Habitat 标准离散动作或连续速度动作，最终报告必须注明具体动作接口。当前 `NarrowPassageNav-v0` 使用连续 `velocity_control`：

```text
a_t = [v_x, w_z]
```

| 参数 | 当前配置 |
|---|---:|
| Linear velocity `v_x` | `[-0.15, 0.35] m/s` |
| Angular velocity `w_z` | `[-45, 45] deg/s` |
| Control time step | `0.25 s` |

PPO、SAC 和 TD3 direct-control baseline 直接输出 `v_x, w_z`。DEGNAV-Rule 先选择 Commit、Explore、Recover 或 Reject，再由共享 mode-conditioned controller 生成连续速度。DEGNAV-RL 仅学习高层 mode selector，但当前结果为诊断性负结果。

### 5.11.5 Training Setting

当前 Habitat-Baselines narrow-passage PPO 配置面向局部控制 baseline，而不是完整 PointNav / ObjectNav 主方法。主要设置如下：

| Hyperparameter | Value |
|---|---:|
| Number of environments | 8 |
| Total training steps | `1.0e7` |
| Rollout length | 128 |
| PPO epochs | 3 |
| Mini-batches | 4 |
| Learning rate | `1.0e-4` |
| Clip parameter | 0.15 |
| Value loss coefficient | 0.75 |
| Entropy coefficient | 0.01 |
| Maximum gradient norm | 0.3 |
| Discount `gamma` | 0.98 |
| GAE `lambda` / config `tau` | 0.92 |
| Hidden size | 256 |
| Maximum episode steps | 400 |
| Depth resolution in narrow PPO config | `128 x 128` |
| Checkpoint interval | 500 updates |

训练 reward 由 progress、clearance、collision、stuck 和 oscillation 项组成。当前 PPO 专用配置使用：

```text
progress_weight     = 5.0
clearance_weight    = 0.2
collision_penalty   = 6.0
stuck_penalty       = 2.0
oscillation_weight  = 0.25
```

复现实验时应同时记录代码 commit、Habitat/Habitat-Sim 版本、HM3D split、episode IDs、随机 seed、训练步数、checkpoint 和 reward override。若 PointNav 或 ObjectNav 使用不同观测、动作或训练预算，必须单独成表，不能与 Narrow Passage policy 进行不加说明的数值排序。

### 5.11.6 Evaluation Metrics

#### 通用导航指标

| Metric | Definition | Applicable tasks |
|---|---|---|
| Success Rate (SR) | episode 满足任务成功条件的比例 | PointNav / ObjectNav / Narrow Passage |
| SPL | 成功率按最短路径与实际路径比加权 | PointNav / ObjectNav；Narrow Passage 需确保路径日志完整 |
| Distance-to-Goal | episode 结束时到目标的距离 | PointNav / ObjectNav |
| Episode Steps | 完成或失败所需控制步数 | 全部任务 |

#### 窄通道专用指标

| Metric | Definition | Interpretation |
|---|---|---|
| Collision Rate | episode 出现 collision flag 的比例 | 执行失败指标 |
| Timeout / Stuck Rate | 超出预算或低进度卡滞比例 | 与 Reject 分开报告 |
| Explicit Reject | 控制器主动选择 Reject | 不能由 `success == 0` 推导 |
| Correct Reject | benchmark 标注不可行且显式 Reject | False-feasible 核心指标 |
| False Reject | 可行 passage 被显式 Reject | Memory interference 指标 |
| Near-collision Rate | depth-derived body margin 低于阈值 | Clearance-aware diagnostic |
| Clearance-aware Strict SR | Success 且无 collision/stuck，并满足 clearance proxy | Diagnostic，不是物理安全率 |
| Minimum Clearance | episode 内最小 depth-derived body margin | 可能受扫描场景、navmesh 和深度伪影影响 |

Habitat clearance-related metrics 来自深度观测和近似机器人 body-margin 模型，用作 clearance-aware diagnostic indicators，而不是经过标定的真实物理安全测量。

### 5.11.7 Local Results and Result Analysis Template

#### 已完成：PointNav / ObjectNav pipeline checks

| Task | Dataset / Protocol | Episodes | Checkpoint | Success | SPL | SoftSPL | Final distance | Evidence role |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PointNav PPO | Habitat test scenes / val | 50 | 981,536 frames | 94.0% | 0.792 | N/A | 0.114 m | Platform validation / diagnostic |
| ObjectNav DD-PPO | HM3D ObjectNav v2 / `val_mini` artifact | 20 | 17,113,408 frames | 15.0% | 0.113 | 0.640 | 1.074 m | Smoke / pipeline diagnostic |

上述两行证明标准 Habitat task pipeline 已在本地执行，但不构成跨任务排行榜。PointNav 使用 example config 且只有单 checkpoint；ObjectNav 是单次 `val_mini` smoke run。二者均不与 `NarrowPassageNav-v0` 的 same-split canonical comparison 混排。

#### 已完成：HM3D same-split nominal anchor validation

| Method | Success | Average steps | Episodes | Evidence status |
|---|---:|---:|---:|---|
| PPO direct-control baseline | 10.2% | 498.0 | 157 | 已完成，同一 HM3D Val set A |
| APF+Gap | 93.6% | 111.5 | 157 | 已完成，同一 HM3D Val set A |
| DEGNAV-Rule | 100.0% | 54.0 | 157 | Nominal anchor validation |
| DEGNAV-Rule + failure memory | 100.0% | 54.0 | 157 | One-shot anchors 上无额外收益 |

可用于论文的保守分析：

> 在当前 HM3D Val set A 的 157 个相同 episode IDs 上，DEGNAV-Rule 在 nominal mined anchors 中取得稳定通过结果，并显著减少平均执行步数。该结果说明显式几何对齐适用于当前 anchor protocol，但不构成对任意 HM3D 场景、任意初始状态或真实四足机器人安全性的普遍证明。Failure memory 在 one-shot passable anchors 上不显示增益，其贡献应由 repeated false-feasible protocol 评估。

#### 已完成：Habitat perturbation sensitivity

| Setting | Key comparison | Measured result |
|---|---|---|
| Nominal | APF+Gap vs DEGNAV full | 96.7% vs 100.0% SR |
| Yaw `60 deg` | Full vs w/o heading alignment | 100.0% vs 76.8% SR |
| Extreme narrow + yaw `60 deg` | Full vs w/o heading alignment | 100.0% vs 79.2% SR |
| Lateral offset `0.20 m` | DEGNAV full | 98.0% SR |

可用于论文的保守分析：

> Controlled yaw and lateral perturbations expose module sensitivity that is hidden by the nominal anchors. The degradation of the no-heading-alignment variant under yaw perturbation supports the role of explicit entrance alignment. These experiments measure sensitivity under the tested HM3D slices; they do not establish universal robustness.

#### 待完成：PointNav / ObjectNav 正式泛化实验

| Task | Required experiment | Required outputs | Current status |
|---|---|---|---|
| PointNav | HM3D seen/unseen，多 seed PPO baseline | Per-episode CSV、SR、SPL、distance、video | 已有 test-scenes pipeline result；HM3D formal result 待运行 |
| ObjectNav | HM3D category navigation，多 seed baseline | SR、SPL、SoftSPL、goal distance、failure cases | 已有 `val_mini` smoke result；held-out multi-seed result 待运行 |
| VLN-conditioned navigation | Language instruction to waypoint | nDTW / SDTW、SR、SPL、grounding error | 未来集成 |

结果分析模板：

```text
On [split] with [N] episodes and [S] random seeds, [method] obtains
[SR] success and [SPL] SPL. Relative to [baseline], the main improvement
appears in [scene/task subset]. Failure decomposition shows that [fraction]
of unsuccessful episodes are caused by semantic localization, geometric
infeasibility, collision, timeout, or control instability. Because [known
limitation], this result is interpreted as [bounded claim] rather than
[overclaim].
```

### 5.11.8 Visualization Suggestions

建议在飞书和论文中按以下顺序组织 Habitat 可视化：

1. **环境与任务图**：HM3D RGB、depth、top-down map，以及 PointNav、ObjectNav、Narrow Passage 三类目标。
2. **观测空间图**：RGB、depth、point goal 与 19-D geometry feature 的对应关系。
3. **轨迹对比图**：在相同 episode 上绘制 PPO direct control、APF+Gap 和 DEGNAV-Rule。
4. **关键帧序列**：Start、before entry、inside/recovery、success/timeout。
5. **扰动敏感性图**：nominal、yaw60、extreme-yaw60 和 lateral-offset 的方法对比。
6. **失败分解图**：Collision、timeout/stuck、explicit reject 和 success 分开显示。
7. **诊断图**：depth-derived body margin、near collision 和 strict success 单独成图，并在 caption 中写明 proxy 限制。
8. **视频素材**：同一 HM3D episode 的 PPO 旋转/超时、APF+Gap 和 DEGNAV-Rule 通过对照。

# 第六部分 Failure-aware Memory

详细 Method 文档：`examples/narrow_passage_rl/docs/failure_aware_memory_method.md`

## 6.1 研究目标

Failure Memory 的目标不是提高 one-shot nominal passable-anchor success，而是减少机器人对历史不可行或 false-feasible 通道的重复 commitment、浪费步数和重复碰撞。

## 6.2 方法介绍

每个 memory item 可记录：

```text
{
  geometry_embedding,
  scene_id,
  passage_width,
  width_body_ratio,
  entrance_angle,
  clearance_asymmetry,
  action_mode,
  outcome,
  min_clearance,
  oscillation_count,
  final_pose_error
}
```

几何约束相似度可写为：

```text
sim =
  w1 * cosine(z_current, z_memory)
  - w2 * |width_ratio_current - width_ratio_memory|
  - w3 * |entrance_angle_current - entrance_angle_memory|
  - w4 * |clearance_asymmetry_current - clearance_asymmetry_memory|
```

检索到的失败案例用于估计 memory risk，并影响 Explore、Recover 和 Reject 的模式决策。

## 6.3 实验设计

Memory 实验必须区分：

1. One-shot passable / false-feasible。
2. Same-passage repeated false-feasible。
3. Similar-but-new false-feasible transfer。
4. Similar-but-feasible interference。

比较方法包括 no memory、local intra-episode memory、kNN failure memory、vanilla episodic memory 和 geometry-guided cross-episode memory。

## 6.4 数据和指标

| 指标 | 含义 |
|---|---|
| Passable SR | 可行通道成功率 |
| Passable false reject | 对可行通道的错误拒绝 |
| False-feasible reject | 对不可行通道的显式拒绝 |
| Transfer reject | 对相似新不可行通道的迁移拒绝 |
| Interference false reject | 对相似可行通道的干扰性错误拒绝 |
| Wasted attempts / steps | 未正确拒绝时浪费的尝试或步数 |
| Retrieval precision | 检索并拒绝案例中真实不可行的比例 |

## 6.5 One-shot false-feasible outcome decomposition

| Method | Traversal success | Explicit reject | Correct reject | Collision | Near collision | Timeout/stuck | Wasted attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reactive rule | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 100.0±0.0% | 57.6±2.6% | 0.0±0.0% | 100.0±0.0% |
| DEGNAV-Rule | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 2.8±0.7% | 14.5±1.0% | 97.2±0.7% | 100.0±0.0% |

DEGNAV-Rule 降低了 hard collision，但 one-shot correct rejection 仍然很弱。大量 timeout/stuck 不能解释为安全拒绝，`success == false` 也不能推导出 Reject。

## 6.6 Memory transfer and interference

| Method | Passable SR | Transfer reject on new FF | Interference false reject |
|---|---:|---:|---:|
| No memory | 90.0±2.2% | 0.0±0.0% | 0.0±0.0% |
| Vanilla episodic memory | 0.0±0.0% | 100.0±0.0% | 100.0±0.0% |
| kNN failure memory | 0.0±0.0% | 100.0±0.0% | 100.0±0.0% |
| Geometry-guided memory | 90.0±2.2% | 93.0±8.6% | 0.0±0.0% |

Geometry-guided memory 在当前 synthetic memory protocol 中保持 passable success，同时迁移拒绝相似新 false-feasible passage，并避免对相似可行通道产生 false reject。该结论只适用于当前合成协议，不建立广泛真实环境泛化。

## 6.7 图片建议

- Failure memory 写入、检索和风险融合流程图。
- Same-passage recurrence 的多轮轨迹。
- Similar-new transfer 与 similar-feasible interference 对比。
- Memory transfer / interference 四面板图。
- 典型 memory item 和相似度分解示意图。

## 6.8 结果分析

Memory 的主张应写为：“Failure memory suppresses repeated commitments to previously failed infeasible passages and reduces wasted attempts.” 不应写为“Memory 提高 one-shot Habitat success”。

# 第七部分 强化学习与 Isaac Sim 四足运动控制

Reward 设计与 reward-hacking 分析：`examples/narrow_passage_rl/docs/quadruped_rl_reward_design.md`

## 7.1 Chapter Status

本节为 Isaac Sim / Isaac Lab 四足机器人强化学习实验模板。用户报告已开展或完成 RL 训练，但当前仓库尚未归档可核验的 Isaac checkpoint、训练曲线、多 seed CSV 和最终配置。因此，机器人型号、训练预算与结果均保留为待填写项；正式论文提交前必须由实际日志替换。

## 7.2 Simulation Platform

实验平台采用 NVIDIA Isaac Sim 与 Isaac Lab。Isaac Sim 提供刚体动力学、接触、传感器和渲染，Isaac Lab 提供并行环境、强化学习任务、域随机化和训练接口。

| Item | Configuration |
|---|---|
| Isaac Sim version | `[待填写]` |
| Isaac Lab version / commit | `[待填写]` |
| Physics backend | PhysX |
| Simulation time step | `[待填写] s` |
| Control decimation | `[待填写]` |
| Policy frequency | `[待填写] Hz` |
| Parallel environments | `[待填写]` |
| GPU / CUDA | `[待填写]` |
| RL library | `[待填写，例如 RSL-RL / SB3 / rl_games]` |
| Training seeds | `[待填写，建议不少于 3]` |

论文式描述模板：

> We train the quadruped locomotion policy in Isaac Lab using `[N]` parallel environments and a control frequency of `[F]` Hz. The simulator runs with a physics time step of `[dt]` s and a control decimation of `[D]`. All reported results are averaged over `[S]` random seeds.

## 7.3 Quadruped Robot Model

> 本小节按要求留白，待补充实际训练机器人。

| Item | Value |
|---|---|
| Robot name / URDF | `[待填写]` |
| Number of actuated joints | `[待填写]` |
| Body length / width / height | `[待填写]` |
| Total mass | `[待填写] kg` |
| Nominal standing height | `[待填写] m` |
| Joint position limits | `[待填写]` |
| Joint velocity limits | `[待填写]` |
| Torque limits | `[待填写]` |
| Foot geometry / friction | `[待填写]` |
| Actuator model | `[待填写]` |
| Observation latency | `[待填写] ms` |

需要上传：

- 机器人正面、侧面和关节编号图。
- URDF / USD 文件路径。
- 机身与腿部碰撞包络。
- 质量、惯量和执行器参数来源。

## 7.4 Task Definition

目标是训练 command-conditioned quadruped locomotion policy：

```text
pi_low(a_t | v_cmd, w_cmd, IMU, joint_state, contact, local_geometry)
```

策略接收高层期望线速度、角速度和可选姿态命令，输出关节位置目标、PD residual、力矩或足端轨迹。实验应覆盖：

1. 平地前进、后退和转向。
2. 低速窄通道跟踪。
3. 入口 yaw alignment。
4. 非对称 clearance 下的姿态保持。
5. 碰撞或卡滞后的倒退恢复。
6. 外力、低摩擦和控制时延扰动。

## 7.5 Observation and Action Space

### Observation

```text
o_t = [
  command,
  base_linear_velocity,
  base_angular_velocity,
  gravity_projection,
  joint_position,
  joint_velocity,
  foot_contact,
  previous_action,
  optional_local_geometry
]
```

| Observation | Dimension / normalization |
|---|---|
| Velocity command | `[待填写]` |
| IMU / gravity projection | `[待填写]` |
| Joint position / velocity | `[待填写]` |
| Foot contact | `[待填写]` |
| Previous action | `[待填写]` |
| Local geometry / clearance | `[待填写或不使用]` |
| Privileged critic observation | `[待填写；部署时不可提供给 actor]` |

### Action

```text
a_t = [joint_position_target]
```

或：

```text
a_t = [joint_torque / PD residual / foot target]
```

最终论文必须注明 action scale、PD gains、action clipping、控制频率和实机接口。

## 7.6 Policy and Training Setup

| Hyperparameter | Value |
|---|---|
| Algorithm | `[待填写，例如 PPO]` |
| Actor / critic architecture | `[待填写]` |
| Recurrent policy | `[是/否]` |
| Total environment steps | `[待填写]` |
| Rollout length | `[待填写]` |
| Batch size | `[待填写]` |
| Learning rate | `[待填写]` |
| Discount factor | `[待填写]` |
| GAE lambda | `[待填写]` |
| Entropy coefficient | `[待填写]` |
| Checkpoint interval | `[待填写]` |
| Best-model selection metric | `[待填写]` |

建议 curriculum：

| Stage | Training content |
|---|---|
| 1 | 平地速度跟踪与站立稳定 |
| 2 | 转向、低速和横向扰动 |
| 3 | 门槛、坡面与摩擦随机化 |
| 4 | 窄通道和入口姿态约束 |
| 5 | 碰撞、卡滞与恢复初始状态 |
| 6 | 感知噪声、时延和 actuator randomization |
| 7 | 与 DEGNAV mode / waypoint 联调 |

## 7.7 RL Reward Design

总 reward 定义为：

```text
r_t =
    w_tracking * r_tracking
  + w_alive    * r_alive
  + w_forward  * r_forward
  + w_center   * r_center
  - w_collision  * r_collision
  - w_energy     * r_energy
  - w_smooth     * r_smooth
  - w_termination * r_termination
```

所有正项越大越好，所有被减去的项定义为非负代价。

| Term | Role | Design rationale | Main risk |
|---|---|---|---|
| `r_tracking` | 跟踪速度、yaw 和姿态命令 | 连接导航输出与低层运动 | 可能以碰撞换跟踪 |
| `r_alive` | 保持站立和 episode 存活 | 提供早期稳定学习信号 | 权重过大导致原地站立 |
| `r_forward` | 接近局部目标 | 直接优化任务进展 | 可能切角或穿越障碍 |
| `r_center` | 保持横向居中与 clearance 均衡 | 降低窄空间单侧接触 | 过强会抑制转弯和绕障 |
| `r_collision` | 惩罚机身/非期望腿部接触 | 防止碰撞换进度 | 接触标签错误会破坏步态 |
| `r_energy` | 降低力矩和机械功 | 提高实机可执行性 | 过强导致低速或静止 |
| `r_smooth` | 抑制动作突变和 jerk | 减少抖动与带宽失配 | 可能抑制快速恢复 |
| `r_termination` | 惩罚跌倒、严重碰撞和越界 | 区分不可恢复失败 | reset 条件可能被利用 |

建议将 `r_forward` 定义为目标距离的势函数差，而不是简单奖励正向速度：

```text
r_forward = d_goal(s_t) - d_goal(s_{t+1})
```

建议将 centering 定义为 passage frame 中的横向误差与 clearance balance，并仅在有效 passage region 启用。Collision 必须区分正常足地接触和机身碰撞。

### Reward Hacking Prevention

1. 裁剪 alive、clearance 和 centering 等可持续累积的 dense bonus。
2. 使 success reward 大于等待至超时可累计的总奖励。
3. 单独记录每个 reward term，不只查看 episode return。
4. 使用独立 Success、fall、collision、foot slip 和 energy/m 指标评测。
5. 用 potential-based progress 防止前后往复刷奖励。
6. 将 collision、fall 和 torque limit 作为约束，而不只作为软惩罚。
7. 构造必须倒退才能恢复、中心可等待但必须前进才能成功的 adversarial cases。
8. 分开处理 success termination、failure termination 和 time-limit truncation。

### Improving Paper Novelty

Reward 项数量本身不构成创新。建议强调：

- **Mode-conditioned reward**：Commit 强调 progress，Explore 强调信息增益，Recover 强调 unstuck，Reject 强调 correct abstention。
- **Belief-conditioned risk budget**：根据 `p_feas`、`delta_var` 和 memory risk 调整速度与约束。
- **Morphology-aware clearance**：使用 yaw 与机器人包络相关的 required width。
- **Failure-memory consistency**：惩罚对历史相似不可行 passage 的重复 commitment，同时惩罚对相似可行 passage 的 false reject。
- **Recovery curriculum**：从碰撞后、轻度卡滞和姿态偏差状态专门训练恢复技能。

## 7.8 Baselines

| Baseline | Purpose | Status |
|---|---|---|
| PD / scripted gait | 非学习运动控制参考 | `[待填写]` |
| PPO locomotion without geometry | 普通 command tracking | `[待填写]` |
| PPO + geometry observation | 检查几何输入贡献 | `[待填写]` |
| PPO + geometry + recovery curriculum | 检查 recovery training | `[待填写]` |
| Full mode-conditioned locomotion | 完整低层方案 | `[待填写]` |

所有 baseline 必须使用相同 robot model、terrain、command distribution、训练步数和 seed 数。

## 7.9 Evaluation Metrics

| Metric | Definition |
|---|---|
| Command Tracking RMSE | 实际与目标线/角速度误差 |
| Locomotion Success | 在时间预算内完成指定运动技能 |
| Fall Rate | base height/attitude 超阈值的 episode 比例 |
| Body Collision | 非足端 contact event |
| Recovery Success | 进入恢复初始状态后重新稳定并继续任务 |
| Foot Slip | 接触期间足端切向位移或速度 |
| Roll / Pitch Stability | 姿态绝对值或 RMS |
| Energy per Meter | 成功轨迹机械功除以前进距离 |
| Action Smoothness | action delta / jerk |
| Sim-to-Real Gap | 仿真与实机对应指标差 |

## 7.10 Result Tables to Fill

### Main Locomotion Result

| Method | Tracking RMSE ↓ | Fall ↓ | Collision ↓ | Recovery ↑ | Slip ↓ | Energy/m ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Scripted / PD | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| PPO baseline | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| Full method | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |

### Reward Ablation

| Variant | Success ↑ | Fall ↓ | Collision ↓ | Oscillation ↓ | Energy/m ↓ |
|---|---:|---:|---:|---:|---:|
| Full reward | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| w/o center | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| w/o energy | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| w/o smooth | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| w/o recovery curriculum | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |

## 7.11 Result Analysis Template

```text
The full policy achieves [RESULT] command-tracking error and [RESULT] recovery
success over [N] episodes and [S] seeds. Compared with the PPO locomotion
baseline, the improvement is concentrated in [SCENARIO], where
[GEOMETRY/RECOVERY MECHANISM] reduces [FAILURE TYPE]. Removing [REWARD TERM]
changes [METRIC] from [A] to [B], indicating [BOUNDED INTERPRETATION].
No statistical significance is claimed unless supported by an explicit test.
```

## 7.12 Figures and Videos to Upload

1. Isaac Sim / Isaac Lab training environment overview。
2. Robot joint and contact visualization。
3. Curriculum stages。
4. Per-term reward and success curves for every seed。
5. Tracking error、roll/pitch、foot slip 和 energy 曲线。
6. Push recovery、low-friction 和 latency test videos。
7. Narrow-passage Commit / Explore / Recover skill comparison。

## 7.13 Current Evidence Boundary

当前仓库没有可核验的 Isaac Sim locomotion 多 seed 结果。用户后续应补充 checkpoint、训练配置、TensorBoard/CSV、评测脚本和视频。完成这些材料前，本节是实验 protocol 与 Method 模板，而不是已验证结果章节。

# 第八部分 DeepRobotics Lite3 实机验证

## 8.1 Chapter Status and Objective

目标是在 DeepRobotics Lite3 上验证语言目标、几何决策、Failure Memory 和运动控制的闭环可执行性。用户报告已完成或开展 Lite3 真机验证，但当前仓库尚未归档可核验的重复统计、rosbag 和完整视频，因此以下结果字段保留为待填写。

## 8.2 Hardware Setup

| Component | Configuration |
|---|---|
| Robot | DeepRobotics Lite3 |
| 3D LiDAR | Livox MID360 |
| RGB-D camera | Intel RealSense D435i |
| IMU | Lite3 / D435i / external IMU，`[待确认使用源]` |
| Onboard computer | `[待填写 CPU/GPU/RAM]` |
| Network | `[待填写 Ethernet/Wi-Fi and latency]` |
| Power | `[待填写 battery/runtime]` |
| Emergency stop | `[待填写]` |

传感器标定信息：

| Calibration item | Value |
|---|---|
| `T_base_lidar` | `[待填写]` |
| `T_base_camera` | `[待填写]` |
| Camera intrinsics | `[待填写或文件路径]` |
| LiDAR-IMU time offset | `[待填写]` |
| Sensor frequency | MID360 `[ ] Hz`; D435i `[ ] Hz`; IMU `[ ] Hz` |
| Robot body envelope | `[待填写 length/width/height]` |

硬件图中必须标注传感器安装位置、视场、坐标系、机身包络和计算单元。

## 8.3 Software Architecture

软件基于 ROS2 Foxy 或 Humble。正式实验必须选择实际版本，不能同时写两者：

| Layer | Node / package | Input | Output |
|---|---|---|---|
| Sensor driver | Livox ROS Driver 2 | MID360 packets | `/livox/lidar`, `/livox/imu` |
| RGB-D driver | `realsense2_camera` | D435i | RGB、depth、camera info |
| Localization / mapping | `[待填写 FAST-LIO / SLAM / localization]` | LiDAR、IMU | map/odom pose |
| Global navigation | Nav2 planner | map、goal | global path |
| Local geometry | DEGNAV geometry extractor | depth/point cloud、pose | passage features |
| Failure memory | memory bank | geometry、outcome | memory risk |
| Mode decision | DEGNAV-Rule | geometry、goal、memory | Commit/Explore/Recover/Reject |
| Controller | Nav2 / RL / vendor SDK bridge | mode、velocity | Lite3 command |
| Logger | rosbag2 + experiment logger | all relevant topics | bag、CSV、video |

数据流：

```text
MID360 + D435i + IMU
          |
          v
Localization / 3D Geometry
          |
          v
Nav2 Global Path + Local Passage Features
          |
          v
Failure Memory + Mode Decision
          |
          v
Velocity / Locomotion Command
          |
          v
DeepRobotics Lite3
```

必须记录 ROS domain、QoS、topic frequency、TF tree、端到端 latency、command watchdog 和 emergency-stop 逻辑。

## 8.4 Experimental Scenarios

### Narrow Passage

- 通道宽度：`[待填写] m`。
- Width/body ratio：`[待填写]`。
- 初始 yaw：`[待填写]`。
- 目标：验证 centering、alignment 和 passage traversal。

### Doorway

- 门宽、门框高度和入口角度：`[待填写]`。
- 是否包含门槛：`[待填写]`。
- 目标：验证入口角度与机身包络。

### Furniture Gap

- 家具类型和 gap geometry：`[待填写]`。
- 是否存在桌腿、悬空障碍或非对称 clearance：`[待填写]`。
- 目标：验证非规则空间中的局部几何决策。

### Dynamic Obstacle

- 障碍类型、速度和交互轨迹：`[待填写]`。
- 安全员与停止规则：`[待填写]`。
- 目标：验证暂停、Explore、局部重规划和恢复。

### Failure Recovery

- 初始失败类型：stuck / collision / heading misalignment / blocked passage。
- Recovery trigger 和最大尝试次数：`[待填写]`。
- 目标：验证后退、重对齐、重新规划与人工接管。

建议每个方法在每个场景至少执行 `[待填写，建议 >=10]` 次，并使用配对的起点、目标和障碍布置。

## 8.5 Baselines

| Method | Role | Required configuration |
|---|---|---|
| Nav2 + DWB | 速度采样局部规划基线 | 相同 costmap、速度限制和 global path |
| Nav2 + MPPI | 采样优化控制基线 | 相同 horizon、footprint 和障碍层 |
| Regulated Pure Pursuit | 稳健路径跟踪基线 | 相同 global path |
| DEGNAV-Rule w/o memory | 显式几何控制 | 禁用跨 episode memory |
| DEGNAV-Rule + memory | 完整 failure-aware navigation | memory 跨重复 trial 保留 |
| RL locomotion / local control | 学习式执行基线 | 仅在 checkpoint 与接口已验证时加入 |

TEB 或其他 ROS1 baseline 只有在实际部署并使用同一 protocol 时才能进入主表。

## 8.6 Evaluation Metrics

| Metric | Definition |
|---|---|
| Success Rate | 在时间预算内无人工接管到达目标 |
| SPL / Path Efficiency | 最短可行路径与实际路径长度的比值 |
| Collision Rate | 经人工标注或 contact sensor 确认的非期望接触 |
| Near-collision | 经过标定的真实 clearance 小于阈值 |
| Timeout / Stuck | 超时或固定窗口内有效位移低于阈值 |
| Oscillation Count | 入口附近角速度/heading correction 的重复换向次数 |
| Recovery Success | 触发恢复后重新进入正常导航并成功 |
| Wrong Decision | 可行通道 Reject 或不可行通道重复 Commit |
| Human Intervention | 安全员接管或急停次数 |
| Time to Goal | 从任务开始到成功的时间 |
| Repeated Failure Rate | memory 已记录失败后再次进入同类不可行 passage |
| End-to-end Latency | 感知时间戳到机器人执行 command 的延迟 |

实机 near-collision 阈值必须通过机身尺寸与传感器标定确定，不能直接沿用 Habitat depth-derived proxy。

## 8.7 Experimental Protocol

1. 固定机器人状态、电量范围和速度上限。
2. 为每个 trial 保存唯一 ID、方法、场景、起点、目标和障碍配置。
3. 方法顺序随机化，避免电量、温度和操作者顺序偏差。
4. 每次运行保存 rosbag、外部视频和 episode-level CSV。
5. 人工接管规则在实验前固定。
6. Memory baseline 必须明确 trial 间是否清空。
7. 成功、collision、timeout、stuck 和 Reject 由互不替代的字段记录。
8. 报告绝对次数、比例和置信区间。

## 8.8 Result Tables to Fill

### Main Real-Robot Results

| Method | SR ↑ | SPL ↑ | Collision ↓ | Timeout/Stuck ↓ | Oscillation ↓ | Recovery ↑ | Intervention ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nav2 + DWB | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| Nav2 + MPPI | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| DEGNAV w/o Memory | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| DEGNAV + Memory | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |

### Scenario Breakdown

| Method | Narrow Passage | Doorway | Furniture Gap | Dynamic Obstacle | Failure Recovery |
|---|---:|---:|---:|---:|---:|
| Nav2 baseline | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| DEGNAV w/o Memory | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |
| DEGNAV + Memory | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` |

## 8.9 Result Analysis Template

```text
Across [N] paired real-robot trials, the full system achieves [SR] Success and
[SPL] SPL. Relative to [BASELINE], the improvement is concentrated in
[SCENARIO], where explicit [ALIGNMENT/MEMORY/RECOVERY] reduces [FAILURE MODE].
Removing memory changes repeated-failure rate from [A] to [B] but changes
one-shot passable Success from [C] to [D], indicating that memory primarily
affects recurrence rather than nominal traversal. Collision and intervention
results are interpreted using the pre-registered hardware definitions above.
```

## 8.10 Failure Case: Oscillation at a Narrow Entrance

### Failure Observation

机器人在狭窄通道入口出现连续左右振荡，多次调整入口角度，但没有形成稳定的 forward commitment，最终耗尽时间预算。应从 rosbag 中补充：

- 失败 trial ID：`[待填写]`。
- 持续时间与总 steps：`[待填写]`。
- 角速度换向次数：`[待填写]`。
- heading / lateral error 曲线：`[待填写]`。
- minimum calibrated clearance：`[待填写]`。
- planner/controller mode 序列：`[待填写]`。

### Root Cause

在没有同步日志前，以下仅为待验证假设：

1. Heading alignment 与 obstacle avoidance 目标相互竞争。
2. 局部 costmap 在入口两侧交替更新，导致最优角速度符号频繁变化。
3. 机器人 footprint 或机身包络与真实 Lite3 不一致。
4. D435i depth dropout、MID360 遮挡或 TF/时延引起 clearance 抖动。
5. 控制器缺少 hysteresis 和最小模式驻留时间。
6. 失败后仍重复使用相同局部目标，没有写入或查询 failure memory。

根因只有在角速度、local costmap、geometry feature、TF latency 和 mode 日志对齐后才能确认。

### Limitation of Existing Method

传统 reactive planner 主要根据当前 costmap 重新采样动作，无法区分“需要继续小幅对齐”和“当前策略已经陷入重复振荡”。若没有 passage frame、失败计数和历史检索，相同入口状态会反复触发近似动作。单纯增加障碍膨胀或降低速度可能减少碰撞，但也会提高 timeout 和 false infeasible 判断。

### Our Solution

计划采用以下机制处理该失败：

1. 显式估计 heading error、lateral offset 和 yaw-aware required width。
2. 在进入 Commit 前执行具有 hysteresis 的 Align / Explore。
3. 通过 oscillation detector 识别角速度重复换向。
4. 触发 Recover，执行受限后退和重新对齐。
5. 将 geometry、action sequence 和 timeout outcome 写入 Failure Memory。
6. 再次遇到相似入口时，提高 memory risk，并选择 Explore、Recover 或 Reject。

在完成对应 Lite3 对照实验前，应描述为 proposed mitigation，而不是已经解决的实机结果。

### Future Improvement

1. 融合 MID360 点云与 D435i depth，降低单传感器入口抖动。
2. 使用 3D body envelope 代替纯 2D footprint。
3. 在线估计 perception-to-control latency。
4. 学习 recovery skill，但保留显式触发条件和安全边界。
5. 对 mode threshold、hysteresis 和 memory retrieval 做实机消融。
6. 增加动态障碍下的短期 motion prediction。

## 8.11 Figures and Videos to Upload

1. Lite3 正面、侧面和传感器安装图。
2. MID360、D435i 和 IMU 坐标系。
3. ROS2 node graph、TF tree 和 topic frequency。
4. 五类实验场景及实际尺寸。
5. 同一起点的 Nav2、w/o Memory 与 full trajectory。
6. 窄入口振荡的 RGB-D、point cloud、costmap 和 mode 时间序列。
7. Collision、timeout、Recover 和 successful retry 视频。

## 8.12 Current Evidence Boundary

当前尚无可核验的 Lite3 多次重复统计和完整闭环结果。用户后续应上传 rosbag、外部视频、episode CSV、传感器标定和 baseline 配置。在此之前，正式材料可写为 experiment design 或 pilot protocol，不能写“广泛实机验证”或“完整 sim-to-real 成功”。

# 第九部分 Sim-to-Real 与系统集成

## 9.1 研究目标

统一 Habitat 决策、Isaac 动力学和 Lite3 ROS2 执行接口，使同一高层模式能够在不同平台上以一致语义运行。

## 9.2 统一接口

| 接口 | 建议统一内容 |
|---|---|
| Observation | RGB-D、pose、geometry、IMU、joint、contact |
| Action | mode、`vx`、`wz`、skill command |
| Coordinate | map、odom、base、camera 坐标系 |
| Timing | 感知、决策、控制频率与时间戳 |
| Limits | 速度、角速度、姿态、接触和急停 |
| Logging | episode ID、scene、seed、trajectory、failure status |

## 9.3 实验设计

1. Offline replay validation。
2. Software-in-the-loop。
3. Isaac closed-loop。
4. Lite3 low-speed pilot。
5. 完整导航任务。

## 9.4 数据和指标

记录 observation distribution shift、depth noise、command tracking error、action latency、domain gap、performance drop 和 intervention rate。

## 9.5 图片建议

- Habitat、Isaac Sim、ROS2 和 Lite3 的接口关系图。
- 坐标系与传感器外参示意图。
- 仿真与实机观测分布对比。
- 相同任务的仿真和实机轨迹并排图。

## 9.6 结果分析

迁移失败应拆分为语义感知偏差、几何估计偏差、动力学偏差、控制时延和硬件执行约束，而不能统一归因于模糊的“sim-to-real gap”。

# 第十部分 实验结果分级与论文叙事

## 10.1 主论文证据

| 实验 | 支持的结论 |
|---|---|
| Procedural v2 main | DEGNAV-Rule 优于 Reactive rule |
| Procedural core ablation | Alignment 是当前 benchmark 的主要 measured component |
| Margin phase | 相同估计 margin 下比较 success 与 collision |
| False-feasible decomposition | 失败不等于 Reject；DEGNAV 降低 hard collision |
| Memory transfer / interference | Geometry-guided memory 抑制重复失败并降低干扰 |
| Habitat same-split nominal | 当前 mining protocol 下的 anchor validation |
| Habitat stress | Controlled perturbation 下的 module sensitivity |

## 10.2 诊断性证据

| 实验 | 诊断意义 |
|---|---|
| DEGNAV-RL | 当前 reward / action interface 导致模式塌缩 |
| PPO / SAC / TD3 transfer | Learning-only direct control 在 domain shift 下不稳定 |
| Habitat strict clearance | Nominal success 可能掩盖 clearance proxy 问题 |

## 10.3 不应过度声明的内容

1. Habitat nominal 100% 不等于完整鲁棒性。
2. Depth-derived clearance 不等于真实物理安全。
3. False-feasible traversal 0% 不等于 correct rejection。
4. Memory 不提高 one-shot nominal Habitat success。
5. DEGNAV-RL 当前没有学习出可靠 Recover / Reject。
6. 当前没有 VLN、Isaac Sim 和 Lite3 的完整正式结果。
7. Synthetic memory transfer 不建立广泛真实环境泛化。

# 第十一部分 项目风险与解决计划

| 风险 | 当前表现 | 解决计划 |
|---|---|---|
| Reject 未触发 | One-shot correct reject 为 0% | 增加显式 infeasible 标签、拒绝监督和代价 |
| Recovery 缺少区分 | nominal ablation 中 w/o recovery 与 full 相同 | 增加卡滞、碰撞后恢复和动态扰动场景 |
| Belief / yaw prior 未充分隔离 | deterministic/no-yaw 与 full 接近 | 扩大 near-boundary 与 forced-yaw 样本 |
| Habitat clearance proxy 异常 | near-collision 高、margin 可为负 | 校准传感器、navmesh 和机身模型 |
| DEGNAV-RL 模式塌缩 | Recover / Reject 为 0 | 模式监督、分层 curriculum、离线示范 |
| VLN 缺结果 | 无标准指标 | 建立 seen/unseen 指令数据与基线 |
| Isaac Sim 缺结果 | 无多 seed 曲线 | 固定训练预算与评测脚本 |
| Lite3 缺统计 | 只有系统计划 | 建立实验 protocol、rosbag 和视频归档 |

# 第十二部分 时间计划模板

| 阶段 | 任务 | 输出 | 状态 |
|---|---|---|---|
| Phase 1 | Habitat / procedural benchmark | 主表、消融、stress、视频 | 已完成 |
| Phase 2 | Failure memory | recurrence、transfer、interference | 已完成 |
| Phase 3 | VLN 接口 | semantic goal、waypoint、seen/unseen 指标 | 进行中 |
| Phase 4 | Isaac Sim locomotion | checkpoint、曲线、扰动结果 | 用户报告完成，待归档 |
| Phase 5 | ROS2 / Lite3 集成 | 闭环节点、rosbag、pilot video | 用户报告完成，待归档 |
| Phase 6 | 实机正式评测 | 多场景、多重复统计 | 未来计划 |
| Phase 7 | 论文与 artifact | 论文图表、配置、复现脚本 | 未来计划 |

# 第十三部分 飞书素材上传清单

## 13.1 必需图片

1. 总体系统架构图。
2. 三层数据流图。
3. Yaw-aware required-width 图。
4. Rule vs DEGNAV margin-phase 图。
5. Procedural 七类场景图。
6. Habitat RGB / depth / top-down trajectory。
7. Memory transfer / interference 图。
8. Isaac Sim 训练场景与课程阶段图。
9. Lite3 传感器安装与实验场地图。
10. 2D costmap、点云、3D occupancy 与 Geometry Token 的同场景对比图。

## 13.2 必需表格

1. Procedural v2 main benchmark。
2. Procedural core ablation。
3. False-feasible outcome decomposition。
4. Memory transfer / interference。
5. Habitat same-split nominal anchor validation。
6. Habitat stress key slices。
7. DEGNAV-RL diagnostic。
8. Isaac locomotion 多 seed 结果，待生成。
9. Lite3 实机重复实验结果，待生成。

## 13.3 必需视频

1. VLN 指令到目标生成，待录制。
2. DEGNAV-Rule nominal narrow passage。
3. Yaw60 入口对齐与 Commit。
4. Reactive rule 与 DEGNAV-Rule 对照。
5. False-feasible collision / timeout / Reject 对照。
6. Memory 避免重复失败。
7. Isaac Sim locomotion 与扰动恢复，待录制。
8. Lite3 实机窄通道和恢复，待录制。
9. RGB-D 点云与 3D occupancy 在线构建，待实现后录制。

## 13.4 复现材料

1. Python、Habitat-Lab、Habitat-Sim、CUDA 与 GPU 版本。
2. Dataset split、episode IDs 和 scene IDs。
3. 随机 seed 和训练预算。
4. YAML / JSON 配置。
5. Checkpoint。
6. Episode-level CSV。
7. 图表生成命令。
8. ROS2 launch、rosbag 和实机参数。

# 第十四部分 仓库结果与文档索引

## 14.1 主要结果表

- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/report_table_all_experiments.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/report_table_habitat_general_navigation.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_procedural_v2_main.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_procedural_v2_ablation_core.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_false_feasible_outcomes.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_same_split.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_key_slices.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_degnav_rl_diagnostic.md`

## 14.2 方法与实验协议

- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/failure_aware_memory_method.md`
- `examples/narrow_passage_rl/docs/quadruped_rl_reward_design.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/habitat_stress_validation.md`
- `examples/narrow_passage_rl/docs/results_registry.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`

## 14.3 关键复现命令

```bash
# Procedural v2 主结果
python examples/narrow_passage_rl/eval_harder_benchmark.py \
  --methods rule_baseline geometry_fsm \
  --episodes 500 \
  --seeds 42 43 44

# 核心消融
python examples/narrow_passage_rl/eval_harder_benchmark.py \
  --variants full no_alignment no_recovery deterministic_margin no_yaw_prior \
  --episodes 500 \
  --seeds 0 1 2 \
  --log-belief-diagnostics \
  --log-outcome-decomposition \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/procedural_ablation_core.csv

# Habitat stress
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
  --preset paper \
  --split val \
  --num-episodes -1

# Memory transfer / interference
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
  --preset paper

# 重新生成论文表格
python examples/narrow_passage_rl/make_paper_tables.py
```

# 第十五部分 汇报页排版建议

## 15.1 首页

放置项目名称、三层架构图、六项研究内容、阶段状态和一句核心结论。

建议核心结论：

> 当前证据表明，在接近几何可行性边界的窄通道中，显式几何对齐与模式决策比 reactive rule 更稳定；failure memory 的主要价值是减少重复不可行 commitment，而 VLN、四足 RL locomotion 和 Lite3 闭环仍处于系统集成与正式评测阶段。

## 15.2 方法页

依次展示 VLN、Geometry / Belief、Failure Memory、Mode Decision 和 RL Locomotion。每个模块只写输入、输出、核心公式和一张示意图。

## 15.3 实验页

按照以下顺序：

1. Procedural main benchmark。
2. Core ablation。
3. Margin phase。
4. False-feasible decomposition。
5. Memory transfer / interference。
6. Habitat nominal 与 stress。
7. DEGNAV-RL negative diagnostic。
8. Isaac Sim 与 Lite3 进行中工作。

## 15.4 结论页

结论应区分：

- 已被当前实验支持的结论。
- 诊断性负结果。
- 尚待验证的 VLN、locomotion 和实机贡献。

避免使用“完全鲁棒”“完美泛化”“解决 HM3D”“安全通过所有窄通道”等超出当前证据的表述。

# 第十六部分 Future Work：面向四足机器人的 3D 导航

## 16.1 Research Motivation：为什么 2D 导航不足

当前移动机器人导航系统通常将三维环境投影到二维 occupancy grid 或 costmap，并在平面上规划机器人质心轨迹。该表示对规则地面和轮式机器人具有较高工程效率，但在真实室内环境中会丢失决定四足机器人可通行性的垂直结构与姿态约束。

1. **高度信息投影丢失**：桌面、桌腿、柜体、门框横梁和悬空障碍在二维投影后可能被合并为同一占据区域，无法区分“可从下方通过”“机身会碰撞”与“足端可跨越”。
2. **机身与足端约束不同**：四足机器人不是平面圆盘。机身、腿部摆动空间和足端落点具有不同包络，单一 2D footprint 难以同时描述 body collision 与 foothold feasibility。
3. **地形高低与姿态耦合**：台阶、坡面、门槛和不规则地面会改变 roll、pitch、支撑多边形和质心稳定性，而 2D costmap 只能表达平面占据代价。
4. **遮挡与未知空间**：RGB-D 在家具密集环境中产生部分观测，二维投影容易把未知区域当作可行或不可行，难以显式保留垂直方向的不确定性。
5. **非规则窄空间**：门框、桌下空间和家具间隙的有效宽度随高度、yaw 和机身姿态变化。固定 2D inflation radius 会过度保守或低估碰撞风险。
6. **动态环境表达有限**：行人、开合门和移动家具同时具有三维形状与速度，仅在平面 costmap 中更新障碍中心难以表达未来占据范围。

因此，未来系统应从“二维质心路径规划”升级为“语言与视觉条件下的三维可通行性推理”，同时保留 2D Nav2 作为全局拓扑导航和工程回退接口。

## 16.2 Proposed 3D Spatial Representation

### 16.2.1 Point Cloud

由深度图、相机内参和机器人位姿将像素反投影为局部点云：

```text
P_t = BackProject(Depth_t, K, T_world_camera)
```

连续时刻点云通过位姿对齐、体素下采样、离群点过滤和时间衰减进行融合。局部点云保留桌面高度、门框形状、地面法向和障碍物垂直轮廓，是后续 occupancy 与 geometry token 的基础。

建议处理流程：

```text
RGB-D
  -> depth validity mask
  -> camera-frame point cloud
  -> robot/world-frame transform
  -> voxel downsampling
  -> temporal fusion
  -> ground / obstacle / overhang separation
```

### 16.2.2 RGB-D Fusion

RGB 提供物体类别、材质、边界和动态目标线索，Depth 提供尺度与几何。未来方法不直接将二者简单拼接，而是在统一三维坐标中为每个点或体素关联：

```text
[position, color/visual embedding, occupancy,
 surface normal, semantic score, timestamp, uncertainty]
```

语义信息用于区分门、桌子、椅子和行人；几何信息用于判断机器人是否能从其旁边、下方或上方通过。两者共同决定局部 waypoint 是否在语义上正确且在物理上可执行。

### 16.2.3 3D Occupancy

局部三维地图可采用 probabilistic voxel occupancy、TSDF / ESDF 或稀疏 voxel hash。每个体素至少保存 free、occupied 和 unknown 三类概率，并记录观测置信度：

```text
O_t(x, y, z) = [p_free, p_occupied, p_unknown, confidence]
```

从 3D occupancy 可进一步提取：

- 机身高度范围内的 body clearance。
- 腿部摆动范围内的 leg-envelope clearance。
- 可落足区域与 surface normal。
- 台阶高度、坡度与 roughness。
- Overhang / under-table clearance。
- 动态物体的短时占据预测。

3D occupancy 不替代所有 2D 模块。系统可投影出多层 2.5D traversability map 供 Nav2 全局规划，同时保留完整 3D 表示供局部几何决策和四足运动控制。

### 16.2.4 Geometry Token

为了与 VLN Transformer 或高层决策网络结合，可将局部点云或 occupancy 划分为若干空间区域，并编码为 Geometry Tokens：

```text
g_i = [
  relative_position,
  local_extent,
  free_width,
  free_height,
  body_clearance,
  foothold_score,
  surface_normal,
  slope,
  occupancy_uncertainty,
  semantic_embedding,
  dynamic_score
]
```

Geometry Token 的核心作用不是替代原始点云，而是将局部三维结构压缩为可解释、可检索和可与语言对齐的中间表示。对于窄通道，可额外加入 yaw-aware required width、entrance angle、left/right asymmetry 和 passage continuation。

### 16.2.5 Spatial Feature

Geometry Tokens 通过局部图网络、稀疏 3D encoder 或 Transformer 聚合为空间特征：

```text
z_3d = SpatialEncoder({g_i}, robot_pose, robot_morphology)
```

`z_3d` 应同时保留：

1. Egocentric local geometry，用于即时控制。
2. Allocentric map context，用于跨房间规划。
3. Robot-conditioned traversability，用于区分“空间存在”与“机器人可通过”。
4. Uncertainty，用于决定 Explore、Recover 或 Reject。

## 16.3 Language-Vision-3D Geometry Fusion

未来系统采用分层跨模态融合，而不是让单一端到端网络直接从语言和图像输出关节动作。

```text
Language Instruction
        |
        v
Language Encoder -> task entities / spatial relations / constraints
        |
        +-------------------------+
                                  v
RGB -> Visual Tokens ------> Cross-modal Fusion
                                  ^
Depth / Point Cloud               |
        -> 3D Occupancy -> Geometry Tokens / Spatial Features
                                  |
                                  v
                 Semantic Goal + 3D-valid Waypoint
```

Language encoder 提取目标实体、房间关系、方向约束和顺序要求；visual tokens 完成开放词汇目标识别与语义定位；geometry tokens 判断候选目标周围是否存在满足机器人形态约束的三维可达区域。融合模块输出的不只是语义目标，还应包含：

```text
{
  semantic_goal,
  candidate_waypoints,
  waypoint_confidence,
  3d_traversability,
  required_posture,
  information_gain,
  uncertainty
}
```

例如，对于“穿过桌子旁边的门到沙发附近”这一指令，语言模块识别门和沙发，视觉模块定位候选实例，三维几何模块排除机身高度不足、门框过窄或落足区域不稳定的候选 waypoint。

## 16.4 Integration with Reinforcement Learning Control

### 16.4.1 Hierarchical Interface

高层策略不直接输出关节动作，而是输出 waypoint、模式与运动技能参数：

```text
High-level state:
  language embedding
  visual-semantic embedding
  3D spatial feature
  feasibility belief
  failure memory

High-level output:
  waypoint
  mode in {Commit, Explore, Recover, Reject}
  locomotion skill
  desired velocity / posture
```

低层 RL locomotion policy 接收：

```text
[desired velocity, desired yaw, body-height command,
 local terrain feature, IMU, joint state, contact, previous action]
```

并输出 joint target、torque 或 foot trajectory。这样，高层负责语义与风险，低层负责动力学可执行性。

### 16.4.2 3D-aware Reward and Constraints

低层训练可加入以下奖励和约束：

```text
r =
  w_v * velocity_tracking
  + w_yaw * yaw_tracking
  + w_goal * waypoint_progress
  + w_foot * foothold_quality
  + w_clear * body_clearance
  + w_stable * attitude_stability
  - w_collision * body_collision
  - w_slip * foot_slip
  - w_energy * energy
  - w_jerk * action_jerk
```

在部署阶段，3D occupancy 和 morphology envelope 可构成 safety filter，对明显违反机身高度、足端落点或姿态约束的动作进行裁剪。该机制应描述为 model-based constraint 或 diagnostic safety filter，只有经过真实传感器和机器人碰撞标定后，才能提出物理安全结论。

### 16.4.3 Training Curriculum

建议采用以下阶段化训练：

| Stage | 训练内容 | 目标 |
|---|---|---|
| 1 | 平地速度跟踪 | 建立稳定基础步态 |
| 2 | 台阶、坡面与不规则地面 | 学习高度变化与接触适应 |
| 3 | 桌腿、门框和悬空障碍 | 学习 body / leg envelope 约束 |
| 4 | 狭窄通道与 forced-yaw entry | 联合姿态、yaw 和 clearance |
| 5 | 遮挡、depth dropout、动态障碍 | 提高部分观测鲁棒性 |
| 6 | 语言条件 waypoint 与失败记忆 | 完成高低层闭环 |
| 7 | 域随机化与 Lite3 参数适配 | Sim-to-real transfer |

## 16.5 Planned Experimental Design

### 研究问题

1. 3D occupancy 是否比 2D costmap 更准确地区分可通过、可跨越、可从下方通过和不可行空间？
2. Geometry Tokens 是否能提高语言目标附近 waypoint 的物理可执行性？
3. 3D-aware high-level mode 是否能减少机身碰撞、卡滞和无效重规划？
4. 低层 RL locomotion 是否能在保持姿态与接触稳定的同时执行三维约束下的局部运动？
5. Habitat / Isaac Sim 中的提升能否迁移到 Lite3？

### 对照方法

| Group | Baseline |
|---|---|
| 2D navigation | Nav2 2D costmap + DWA / MPPI |
| 2.5D navigation | Elevation / traversability map + local controller |
| 3D geometry | Point cloud / occupancy without language |
| Language navigation | VLN waypoint without robot-conditioned geometry |
| Proposed | Language + Vision + 3D Geometry + Failure Memory + RL control |

### 场景设计

1. 桌下可通行与机身过高的成对场景。
2. 不同高度门框和不同 yaw 的入口。
3. 台阶、坡面、门槛和混合高低障碍。
4. 家具密集、遮挡严重和不规则通道。
5. 动态行人或移动障碍。
6. 语义目标正确但几何不可达的 false-feasible goal。

### 评测指标

| 层级 | 指标 |
|---|---|
| 3D mapping | Occupancy IoU、free-space precision/recall、height MAE、map completeness |
| Traversability | Passability F1、foothold precision、body-clearance error、unsafe waypoint rate |
| VLN | SR、SPL、NE、nDTW、SDTW、goal grounding accuracy |
| Navigation | Collision、near-collision diagnostic、stuck、oscillation、time-to-goal |
| Locomotion | Velocity tracking、roll/pitch、foot slip、fall rate、energy、recovery success |
| Memory | Repeated failure rate、wasted attempts、transfer reject、interference false reject |
| Sim-to-real | Success gap、trajectory deviation、latency sensitivity、real/sim clearance error |

正式实验应至少使用 3 个随机 seeds，并固定场景、指令、初始状态与评测预算。Habitat 与 Isaac Sim 的结果应分表报告；真实 Lite3 结果应给出每个条件的试验次数、成功数、失败类型和置信区间。

## 16.6 Image, Table, and Video Suggestions

### 图片

1. 同一家具场景的 2D costmap、点云、3D occupancy 和 geometry tokens 对比。
2. 桌下、门框和台阶场景中的 body / leg envelope。
3. Language、Visual Tokens 与 Geometry Tokens 的跨模态融合图。
4. Habitat semantic waypoint 到 Isaac Sim locomotion command 的数据流。
5. 2D planner 与 3D-aware planner 的失败/成功轨迹对照。

### 表格

1. 2D、2.5D 和 3D 表示的能力与计算开销对比。
2. 有无 Geometry Token 的 VLN waypoint 可执行性消融。
3. 有无 3D occupancy 的碰撞、卡滞与任务成功对比。
4. Habitat、Isaac Sim 和 Lite3 的 sim-to-real gap。

### 视频

1. 点云和 occupancy 在线更新。
2. 机器人判断桌下空间是否满足机身高度。
3. 门框 forced-yaw alignment 与姿态调整。
4. 语言目标正确但几何不可行时的 Explore / Reject。
5. Isaac Sim 与 Lite3 对应场景的并排回放。

## 16.7 Future Work Statement

未来工作将把当前基于深度扇区和二维 body-margin 的局部几何表示扩展为面向四足机器人形态约束的三维空间模型。具体而言，系统将融合 RGB-D 点云、概率 3D occupancy、局部表面与落足信息，并将其压缩为可与语言和视觉语义对齐的 Geometry Tokens。高层模块据此生成语义一致且三维可执行的 waypoint，并结合 failure memory 选择 Commit、Explore、Recover 或 Reject；低层强化学习策略进一步根据 IMU、关节和接触状态实现满足姿态、落足与机身 clearance 约束的运动。

该方向的核心目标不是以更复杂的三维网络替代现有导航栈，而是在保留 Nav2 全局规划、显式几何决策和可解释失败处理的基础上，补足 2D costmap 对高度、悬空障碍、足端接触与四足姿态约束表达不足的问题。所有关于 3D traversability、VLN 增益和 sim-to-real 的结论都需要在后续 Habitat、Isaac Sim 和 Lite3 对齐实验中验证；在取得可追溯结果前，本部分仅作为研究方案与 Future Work。

# 第十七部分 机器人项目总结

## 17.1 Project Scope

项目围绕 `VLN + Failure Memory + RL Quadruped Navigation` 展开，目标是构建从自然语言任务理解、三维/窄通道几何决策、失败经验复用，到四足机器人运动执行的分层自主导航系统。

```text
Language + RGB-D
       |
       v
Semantic Goal / Waypoint
       |
       v
Geometry + Feasibility + Failure Memory
       |
       v
Commit / Explore / Recover / Reject
       |
       v
RL Quadruped Locomotion
       |
       v
DeepRobotics Lite3
```

## 17.2 Evidence Status

“工程流程已运行”和“论文证据已完整”需要分开。当前状态为：

| Work package | Project milestone | Paper-evidence status |
|---|---|---|
| Habitat-Lab navigation | 已完成 | 有 episode CSV、same-split table、stress table 和视频工具 |
| Failure Memory | 已完成设计与 synthetic protocol | 有 recurrence、transfer 和 interference 结果 |
| Isaac Sim RL training | 用户报告已完成 | 待上传 checkpoint、配置、训练曲线和多 seed 评测 |
| Lite3 real-robot validation | 用户报告已完成 | 待上传 rosbag、标定、重复统计和 baseline 对照 |
| VLN integration | 进行中 | 有 NaVILA runtime、640-case diagnostic 和 safety-adapter smoke；尚无标准 VLN SR/SPL/nDTW 结果 |

该表允许项目汇报使用“已完成里程碑”，同时保证论文只引用可追溯数据。

## 17.3 已取得成果

### Habitat-Lab Navigation

1. 建立 `NarrowPassageNav-v0` 与 19-D geometry observation。
2. 建立 procedural v2、HM3D nominal anchor 和 stress validation。
3. DEGNAV-Rule 在 procedural v2 上达到 `70.3±1.4%` Overall SR，Reactive rule 为 `25.4±1.2%`。
4. 在 HM3D same-split nominal anchors 上完成 157-episode 对照；100% 结果仅解释为当前 mining protocol 下的 nominal validation。
5. 完成 yaw、lateral offset、dropout、noise 和 extreme-narrow module-sensitivity 分析。

### Failure-aware Memory

1. 建立 episode-local 与 cross-episode failure memory。
2. 建立 passage fingerprint、`D_min` calibration 和 geometry-guided retrieval。
3. 将 one-shot failure 与 explicit Reject、collision、timeout/stuck 分开。
4. 在 synthetic transfer protocol 中，geometry-guided memory 保持 `90.0±2.2%` passable SR，实现 `93.0±8.6%` similar-new false-feasible transfer reject，并保持 `0.0%` interference false reject。

### Reinforcement Learning

1. 建立 PPO/SAC/TD3 direct-control baseline 与 DEGNAV-RL diagnostic。
2. 识别 nominal reward exploitation 和 mode collapse 等负结果。
3. 完成 mode-conditioned quadruped reward、reward-hacking audit 和 Isaac Lab 实验 protocol。
4. 用户报告已完成 Isaac Sim RL 训练；论文结果待按第七部分模板归档。

### Real-Robot Integration

1. 明确 Lite3 + MID360 + D435i + IMU + ROS2 + Nav2 系统架构。
2. 建立 Narrow Passage、Doorway、Furniture Gap、Dynamic Obstacle 和 Failure Recovery 实验协议。
3. 建立狭窄入口振荡失败案例的日志与根因分析模板。
4. 用户报告已完成 Lite3 验证；正式统计待按第八部分归档。

## 17.4 当前不足

1. **VLN evidence gap**：尚缺语言指令数据集、seen/unseen split、SR、SPL、nDTW 和 grounding error。
2. **Isaac reproducibility gap**：尚缺 robot USD/URDF、reward weights、checkpoint、多 seed 曲线、training budget 和评测 CSV。
3. **Hardware evidence gap**：尚缺传感器外参、TF tree、rosbag、trial-level CSV、基线配置和重复实验。
4. **Memory domain gap**：现有 transfer/interference 证据来自 synthetic protocol，尚未在 HM3D recurrence 或 Lite3 repeated trials 上验证。
5. **Reject limitation**：one-shot false-feasible correct reject 仍弱，不能把 timeout/stuck 描述为安全拒绝。
6. **Recovery limitation**：当前 nominal ablation 没有充分激活 recovery，需专门的 stuck/collision initial states。
7. **Clearance calibration gap**：Habitat body-margin 是 depth-derived diagnostic，不是实机物理 clearance。
8. **3D representation gap**：当前局部几何仍主要来自 depth sectors，尚未完成 3D occupancy 与 Geometry Token。

## 17.5 下一阶段目标

### Priority 1：归档已完成实验

1. 导出 Isaac Sim 的最终 checkpoint、配置、TensorBoard 和 per-seed CSV。
2. 补齐 Lite3 sensor calibration、rosbag、trial metadata 和外部视频。
3. 按相同 protocol 重跑 baseline，生成可比较主表。
4. 将用户报告完成的里程碑转化为 repository-verifiable evidence。

### Priority 2：完成系统闭环

1. 统一 Habitat、Isaac 和 Lite3 的 velocity、mode、坐标系和时间戳。
2. 完成 VLN semantic goal 到 DEGNAV waypoint 的接口。
3. 将 failure memory 接入真实 repeated trials。
4. 将 RL locomotion 接入 Commit、Explore 和 Recover 的 mode-conditioned command。

### Priority 3：补强论文实验

1. PointNav/ObjectNav/VLN seen-unseen baseline。
2. Isaac reward ablation 与 recovery-specific curriculum。
3. Lite3 DWB/MPPI/DEGNAV w/o Memory/DEGNAV + Memory 配对实验。
4. 实机振荡、碰撞、timeout 和 wrong decision failure decomposition。
5. 多 seed、多场景置信区间与统计检验。

## 17.6 长期研究方向

1. **Language-conditioned 3D navigation**：将自然语言、开放词汇视觉和 robot-conditioned 3D geometry 联合建模。
2. **Lifelong failure memory**：在环境变化和机器人形态变化下更新、遗忘和迁移失败经验。
3. **Risk-aware hierarchical learning**：高层学习 mode 和 waypoint，低层学习接触稳定的 locomotion。
4. **Morphology-aware navigation**：显式建模机身、腿部摆动和足端落点的三维包络。
5. **Verified sim-to-real**：通过动力学辨识、感知标定、时延建模和真实对照量化 domain gap。
6. **Human-robot interaction**：在目标不明确或风险过高时请求澄清、重新规划或人工确认。

## 17.7 Project Summary for Supervisor Review

推荐汇报表述：

> 本项目已建立 Habitat-Lab 窄通道导航、显式几何决策和 Failure Memory 的可复现实验链路，并形成 Isaac Sim 四足强化学习与 DeepRobotics Lite3 实机系统的实验框架。当前可核验结果表明，显式入口对齐与 passage-centric geometry 在接近可行性边界的场景中优于 reactive rule，geometry-guided memory 在合成重复失败协议中能够减少不可行通道的重复尝试，同时避免 generic memory 的过度拒绝。用户已报告完成 Isaac Sim 训练和 Lite3 验证，但论文级 checkpoint、训练曲线、rosbag 和多次重复统计仍需归档。下一阶段工作的重点不是继续扩大未经校验的功能范围，而是完成 VLN 接口、统一仿真与实机 protocol，并把已有系统运行转化为可复现、可审查的论文证据。
