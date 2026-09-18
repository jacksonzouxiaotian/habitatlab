# EAGOR 下一阶段：单层在线建图、探索与离散导航实验

> 日期：2026-09-11。本文是新增导航系统的实现与实测报告，不是 EAGOR 原文最终成绩复现。  
> 实验根目录：[online_navigation_20260911](../eagor_outputs/online_navigation_20260911)。  
> 旧对照：[10 场景失败归因](FAILURE_ATTRIBUTION_10SCENES_CN.md)；组会总索引：[README_GROUP_MEETING_CN.md](README_GROUP_MEETING_CN.md)。

## 1. 结论与交付状态

已经实现：累积深度占据地图、footprint 膨胀、可达 frontier、软方向引导、A*、离散动作跟随、碰撞/无进展反馈、无方向探索，以及独立的在线评估分支。没有修改 SH-BF、coefficient update、旋转传播或方向解码。

已经跑通：原有同批 10 个 MP3D 任务中，**GT Direction + Online Planner + Oracle Stop 为 6/10，SPL=0.369，平均阻塞碰撞 0.4**。旧 GT Direction + Direct + Oracle Stop 为 0/10、平均 24.6 次碰撞。新增链路能在部分真实场景出门、绕障、跨房间接近目标；它仍远未达到旧 Navmesh Follower 的 10/10、SPL=0.978。

关键负结果：统一原 Area 停止规则后，**EAGOR=0/10、Grid=0/10、GT Direction=0/10、无方向探索=0/10、Centroid=1/10**。EAGOR、Grid 都有 4 个任务进入过成功范围，却没有完成正确停止。不能将 6/10 写成 EAGOR 非 Oracle 系统的成功率，也不能宣称本次证明了 SH 表示优于 Grid。

全部使用**无先验地图、仿真真值位姿**。普通方向方法的感知仍是已有 **Oracle Semantic**，VLM 调用为 0。尚未验证真实检测器、带噪里程计、实机定位、跨楼层或论文完整 benchmark。

### 汇报可直接使用的两点 weakness

1. **方向正确不等于路径可达。** 旧 GT Direction + Direct/Depth Local 均为 0/10；新增在线地图与路径执行后，保持 GT Direction/Oracle Stop 的诊断变为 6/10。证据支持这一批任务中的规划与执行瓶颈，不证明所有任务失败都由同一原因造成。
2. **接近目标不等于正确结束任务。** 主实验中 EAGOR 和 Grid 各有 4/10 进入成功范围但最终 0/10；两个共同案例只替换为 Oracle Stop 后均成功，且停止前位姿与动作前缀完全一致。原面积阈值与 ObjectNav 成功区域不是同一判据。这是当前复现系统的停止弱点，不能直接推广为论文所有实现的弱点。

## 2. 实验协议与原任务恢复

没有重新采样任务。旧目录没有找到完整的独立 episode manifest，因此从旧 A 组每个 CSV 第 0 帧恢复 scene、episode、起点、yaw，并与原 ObjectNav 数据集逐项验证；数据集补全四元数和目标定义，保存目标 payload SHA256、原 CSV SHA256。

- [episode_manifest.json](../eagor_outputs/online_navigation_20260911/episode_manifest.json)：含完整目标定义，但明确属于 evaluation-only，绝不传给地图/规划器。
- 历史完整配置快照缺失：当前匹配旧日志的 baseline 配置用于重建协议，不能声称具有当时完整环境镜像。
- 每次新运行再次验证起点、四元数、目标类别及目标 payload 哈希，核对动作、机器人、传感器和成功阈值。
- [fairness_checks.json](../eagor_outputs/online_navigation_20260911/analysis/fairness_checks.json)：五个主实验均为同一组 10 个 key、相同 manifest 哈希、相同决策核心源码及运行导航配置。视频/停止方式的明确实验差异不算导航配置差异。

| 项目 | 固定值 |
|---|---|
| 数据 | MP3D ObjectNav v1，同批 10 scene × episode 0 |
| seed / 顺序 | 0；manifest 明确排序，关闭 iterator shuffle |
| 最大动作预算 | 300 步，含转向与 stop，不因失败更换任务 |
| 动作 | move_forward 0.25 m；turn_left/right 15°；stop |
| 机器人 | 圆盘半径 0.18 m，高 0.88 m |
| 主要观测 | 原生 ERP RGB/depth/semantic 512×256；相机高 0.88 m；保留原 perspective 观测用于显示 |
| 位姿 | Habitat simulator GT pose；不是估计里程计 |
| 成功 | Habitat Success：执行 STOP，且到合法 goal viewpoints 的测地距离严格小于 0.1 m |
| 感知 | 现有 Oracle Semantic，各方法每个新观测调用一次，无候选点重复感知 |
| 主停止规则 | 原 Area：visible、confidence≥0.25、likelihood≥0.5 的像素比例≥0.08 |
| 诊断停止规则 | Oracle Stop：评估层仅输出 distance<0.1 m 的布尔结果 |

`area_depth` 旧接口仍可用，但不是本次主结果；没有调整阈值来制造成功。不同方向表示的 confidence 尚未统一校准：相同阈值/代码不等于置信尺度完全一致。无方向探索用共同 Centroid 感知/停止判断，传给规划器的方向强制无效。

### 原 10 任务并不全是单层任务

离线核对旧 Oracle 成功路径与所有合法 goal viewpoints 后发现：

| scene | 旧成功轨迹高度范围 | 合法 viewpoint 与起点最小高度差 |
|---|---:|---:|
| 8194nk5LbLH | 1.882 m | 1.812 m |
| X7HyMhZNoso | 4.656 m | 4.656 m |

这两任务的目标 viewpoint 不在起点同一高度附近，是单层表示的重要适用范围冲突。没有删掉它们，也没有把剩余 8 个任务另包装成新的主 benchmark。旧轨迹高度范围本身不能证明所有可行路线的形状；这里还有 viewpoint 高度的独立证据。EU6、Z6 的旧路径也有约 0.39/0.43 m 起伏，不能预先把所有失败都叫“跨楼层”。数据见 [elevation_audit.json](../eagor_outputs/online_navigation_20260911/analysis/elevation_audit.json)。这些信息只用于事后分析。

## 3. 模块与真值隔离

```text
ERP depth ──深度约定转换──→ 累积占据地图 ←── 当前位姿 / 实际运动 / 碰撞反馈
                                      │
RGB + 既有感知 → EAGOR/Grid/Centroid ─方向→ reachable frontier + soft score
                                      │
                             已知自由空间 A* → 离散跟随 → 动作

环境目标/goal viewpoints/navmesh ──→ 评估指标与报告（不回流普通规划）
GT Direction 诊断 adapter ──→ 仅 bearing
Oracle Stop 诊断 adapter ──→ 仅最终 STOP 布尔值
```

| 模块 | 新文件 / 修改点 | 输入边界与作用 |
|---|---|---|
| Mapper | [online_mapper.py](planning/online_mapper.py) | radial depth、姿态、位置、步数；累积证据/访问历史；不读语义实例 |
| Frontier | [frontier_planner.py](planning/frontier_planner.py) | 在线地图、当前位置、DirectionCue；连通域筛选、冷却、软评分、回退 |
| A* / 连通性 | [grid_path_planner.py](planning/grid_path_planner.py) | 仅在线 safe grid；8 邻域禁止对角切角；线段复检 |
| 执行 | [online_navigation.py](planning/online_navigation.py) | depth、GT pose、方向、上次碰撞；子目标/waypoint/最终停止分离 |
| 评估 | [online_evaluator.py](evaluation/online_evaluator.py) | 持有 Habitat 环境与任务真值，仅在边界外算指标 |
| 接入 | [habitat_evaluator.py](evaluation/habitat_evaluator.py) | 新增 `planning.mode=online_frontier` 分派；旧 Direct/Depth Local/Navmesh 默认行为保留 |
| 显示 | [online_visualization.py](evaluation/online_visualization.py) | 只显示在线地图、轨迹、frontier、规划路径与 RGB，不查询 GT 顶视图 |
| 配置 | [mp3d_online.yaml](../configs/eagor/mp3d_online.yaml) | 与 base、mp3d_failure_attribution 叠加；明确 pose/planning/stop/manifest |

`DirectionCue` 只有 `world_xz / valid / confidence`；方向归一化后参与评分，零向量、非有限数和无效预测都回退探索。GT adapter 丢弃目标距离与实例索引，不能把固定 bearing probes 解释为目标真实位置。探索半径 0.5/1/2/3/4 m 只是候选采样距离，所有候选仍须在已知连通空间。

四个规划文件没有 Habitat 依赖，也不接收 env、目标坐标、viewpoints、语义 mask、目标实例 ID、task distance 或全局地图。没有主动调用 `find_path`、`snap_point`、`get_topdown_view`、`ShortestPathFollower`、`geodesic_distance` 或 `get_metrics`。针对性 AST 检查和接口测试通过。日志的 `planner_privileged_queries=0` 是这项静态/能力边界审计结果，**不是完整运行时污点追踪计数器**。

仿真器内部的动作约束与碰撞依然由环境负责；评估层仍可以查询距离算 SR/SPL。Oracle Semantic 属于特权感知，尽管它不进入 mapper，也不能把整套系统称作“完全无特权 RGB-D”。

## 4. 深度与地图：实际发现和处理

### 4.1 不能把当前原生 ERP depth 直接当 radial depth

用 Habitat-Sim 0.3.3 实际渲染不对称六面盒子：世界 X[-2,3]、Y[-1,4]、Z[-5,6]。本地原生 ERP depth 保留的是 **cubemap face axial depth**，不是 ERP 中心射线的欧氏距离，也不是一张普通针孔图的共同 Z 深度。

对 ERP 单位射线 `u`，本地适配转换为：

```text
radial_depth = native_face_axial_depth / max(abs(u.x), abs(u.y), abs(u.z))
```

理论平面交点与转换深度比较：中位绝对误差 **0.0000524 m**，P99 **0.06462 m**；错误地将原始值作为 radial 的中位误差 **0.33429 m**。残余边缘误差与采样/插值有关。原始 0、NaN、Inf、接近 10 m 裁剪值无效；地图最大使用距离 5 m。适配器是针对已审计本地 native ERP 构建，不应该直接套用于其他 depth backend。旧 depth helper/旧 StopCriterion 未改，避免改写旧对照。

测试源码：[verify_erp_depth_geometry.py](scripts/verify_erp_depth_geometry.py)。最终证据：[native_depth_verified.json](../eagor_outputs/online_navigation_20260911/geometry/native_depth_verified.json)。`geometry` 内早期检查文件保留调试过程，其中空场景测试装置曾产生零深度，不是最终验证通过的结果。

**【组会图 1 插入位置：深度约定与已知几何校验】**

![深度校验](../eagor_outputs/online_navigation_20260911/analysis/depth_geometry_audit.png)

### 4.2 单层累积地图

- 世界坐标 X/Z 水平、Y 向上；ERP 按既有 positive-left 约定展开，经 body→world 旋转，加入 0.88 m 相机高度。
- 分辨率 0.075 m，初始边长 18 m，按 64 cell 扩容；证据、观测时间、访问计数、接触记录及轨迹一起保留，不丢历史回退路径。
- 占据证据 hit=+1.5、miss=-0.45，范围[-3,5]；occupied≥0.6，observed 且 evidence<0 为 free，其余不规划通行。
- 自由射线只在碰撞高度范围内清空；地面容差 0.07 m；高于此容差且低于 0.88 m 的端点记障碍，悬空高处结构不用于证明地面可通行。同帧障碍证据优先于自由证据。
- 障碍 EDT 膨胀：半径 0.18 + safety margin 0.02 + 半个栅格 0.0375 m。未知中心格不可通行。当前圆盘位置由实际占据的测量位姿清出，不由 GT 地图清出。
- 合法起点位于**额外安全余量**内时，允许在 0.45 m 局部范围沿不小于当前净空的已知 free cells 退出；实际半径约束与 occupied cells 不取消。该模式有独立日志与单元测试。

局限：本实现按起点高度固定一个 collision slab，不是多层地图；栅格化圆盘、深度抽样和柱状高度压缩都是近似，不能保证连续几何绝对安全。unknown 检查主要约束中心路径，不等于严格对整个 footprint 的未知体素完成认证。位姿噪声、透明物体、细小障碍、台阶与长程闭环漂移尚未验证。

## 5. Frontier、A* 与执行细节

Frontier 来自 observed-free 邻接 unknown 的边界，过滤存储边缘和少于 3 格的簇；代表点放到已知 free、膨胀安全且从当前位姿连通的邻近格。候选过近（地图路径长度<0.45 m）不选。

评分各项均归一化：

| 项 | 定义 | 权重 |
|---|---|---:|
| gain | cluster_size/(cluster_size+20) | +1.0 |
| cost | known_map_path_m/(known_map_path_m+3) | -0.7 |
| risk | 1-min(clearance_m/0.6,1) | -0.3 |
| revisit | local_visits/(local_visits+5) | -1.2 |
| direction | (1+cos(bearing, candidate_direction))/2；无效时 0 | +0.8 |

Bearing probe 额外加 0.2 固定偏置，候选与所有分量、权重、排序结果记录在 `frontier_scores.json`。失败或到达的子目标在 0.65 m 邻域冷却 35 步，不永久封禁。无候选时尝试已知轨迹节点回退，再做至多 24 步连续观察；耗尽作为失败退出，不冒充 STOP 成功。

连通性用同一 safe grid 的 Dijkstra 树，选定子目标再跑 A*；这不是 navmesh 连通性查询。8 邻域禁止穿对角障碍缝，0.65 m lookahead 的捷径需要线段复检。每个 0.25 m 前进指令重新检查离散朝向下的 swept center segment，不能用连续朝向代替 15° 动作执行。

记录指令长度和实际位移；一次前进位移<0.04 m 触发恢复，接触点来自当前位置和上一朝向，冷却子目标并重规划。延续旧指标：日志 `collision_count` 只数前进位移<1e-4 m 的**完全阻塞**，不统计所有部分滑动接触，不等于 Habitat 所有接触事件。

路径新障碍失效、子目标到达、60 步子目标超时、24 步无位移进展均有处理。最近已走过的路径会裁掉再检查，避免把身后新障碍误判成剩余路径失效。安全绕障转向后承诺执行一次重新检查过的前进，防止立即转回原路径导致左右摆动。

当前代码仍可能在局部区域前进—回退；无位移检测不是完整的“任务进度”检测。`frontier_scores.json` 中 reason 是最近触发原因，可跨多个无候选轮次延续；**不能把 reason 标签频次当成新的路径失效次数**，应读取 `invalid_paths` 累计计数。

## 6. 验证记录与迭代

| 检查 | 实际结果 |
|---|---|
| 修改前回归 | 原 53/53 通过 |
| 最终回归 | **68/68 通过**，其中新增 15 项 online 测试 |
| 编译 | `compileall -q eagor_repro` 通过 |
| MP3D preflight | `ready=true`，90 场景可见 |
| 原数学等模块 | spherical/policies/controllers/sensors/perception 共 23 个 .py 与首次诊断快照字节一致 |
| 空旷闭环 fixture | 26 步成功，0 碰撞 |
| 单障碍 fixture | 44 步成功，0 碰撞 |
| 门洞隔墙 fixture | 83 步成功，0 碰撞 |
| 必须先背离目标 fixture | 90 步成功，0 碰撞 |

人工布局由独立射线/运动引擎生成 depth，真实矩形只存在于测试环境，不给普通规划器。fixture 使用 128×64 depth 和 0.3 m 测试到达半径，**不进入 MP3D SR，不替代真实视频，也不改变 MP3D 0.1 m 标准**。

新增测试覆盖前后左右/高度/米制、深度转换与裁剪、低障碍和悬空结构、ray carve、历史扩容、宽门/窄门 footprint、unknown、不可达 frontier、A* 绕障与切角、软方向背离、冷却、碰撞/停滞、有界探索耗尽、离散转向衔接前进以及特权输入边界。人工场景脚本在失败/碰撞时返回断言错误。

证据：[verification_release](../eagor_outputs/online_navigation_20260911/verification_release)、[layouts_release](../eagor_outputs/online_navigation_20260911/layouts_release)。前者同时保存交付版本源码/报告快照。旧 53 项测试没有删改成更宽松条件。23 文件哈希对照起点是首次诊断源码快照，不是假称一个不存在的干净 git commit。

首轮 `diagnostic_v1` 是 2/10，修复合法起点安全带退出、离散转向摆动及已走路径裁剪后，固定版 `diagnostic_final` 为 6/10。不是所有任务都单调改善，例如 2az 路径从约 9.75 m 变为 20 m。保留全部版本，不挑每个 scene 最好的一次拼成绩。

## 7. 原 10 任务诊断结果

条件：**GT Direction + Online Planner + Oracle Stop**。下表全部为同一个固定版本。

| scene / episode 0 | 目标 | 成功 | 步数 | 路径 m | 阻塞碰撞 | 最终任务距离 m |
|---|---|---:|---:|---:|---:|---:|
| 2azQ1b91cZZ | cabinet | 是 | 126 | 20.00 | 0 | 0.041 |
| 8194nk5LbLH | gym_equipment | 否 | 300 | 36.00 | 0 | 13.824 |
| EU6Fwq7SyZv | cabinet | 否 | 300 | 15.50 | 0 | 10.257 |
| QUCTc6BB5sX | chest_of_drawers | 是 | 130 | 15.00 | 0 | 0.062 |
| TbHJrupSAjP | counter | 是 | 130 | 9.66 | 2 | 0.084 |
| X7HyMhZNoso | bed | 否 | 300 | 45.32 | 1 | 27.686 |
| Z6MFQCViBuw | chest_of_drawers | 是 | 267 | 31.65 | 1 | 0.051 |
| oLBMNvg9in8 | cabinet | 是 | 168 | 14.25 | 0 | 0.058 |
| pLe4wQe7qrG | shower | 是 | 86 | 8.51 | 0 | 0.060 |
| x8F5xyUWy9e | plant | 否 | 34 | 0.25 | 0 | 2.944 |

| 规划条件 | SR | SPL | 平均阻塞碰撞 | 证据性质 |
|---|---:|---:|---:|---|
| 旧 GT Direct + Oracle Stop | 0/10 | 0 | 24.6 | 上轮留存结果 |
| 旧 GT Depth Local | 0/10 | 0 | 24.7 | 上轮留存结果 |
| 新 GT Online + Oracle Stop | 6/10 | 0.369 | 0.4 | 本轮固定版实跑 |
| 旧 GT Navmesh Follower | 10/10 | 0.978 | 0 | 特权路径参考，未修改旧实现 |

**【组会图 2 插入位置：同批任务在线地图与轨迹，绿轨迹、蓝起点、红终点】**

![十任务诊断地图](../eagor_outputs/online_navigation_20260911/analysis/diagnostic_maps.png)

### 剩余四个诊断失败：证据与未解决处

- **8194**：300 步、36 m、0 阻塞碰撞，35 次真实路径失效、38 次重规划，未进入成功范围。目标 viewpoints 至少高 1.812 m；目前保持单层高度。映射连通性抖动也存在，不能只用一个标签解释全部过程。
- **X7**：300 步、45.32 m、1 次碰撞；合法目标 viewpoints 高差≥4.656 m。单层地图没有楼梯/楼层拓扑，不能完成这种能力；本轮不补跨楼层规则。
- **EU6**：300 步、15.50 m、0 阻塞碰撞，124 次重规划、13 次真实路径失效、9 次 backtrack 选择。视频显示在柜台/家具附近反复观察与回退，没有持续推进到目标房间。终止类别为 budget exhausted；地图碎片化、保守连通性与 frontier 选择之间的具体主因仍标记 unresolved，未用 GT 地图得出虚假的精确归因。
- **x8**：只前进一次 0.25 m；第 9 步地图更新后剩余路径失效，此后虽有约 263 个 raw frontier cells，却没有合格可达候选；连续观察后第 34 条日志退出。raw frontier 数量不等于可达 frontier 数量。更可能是地图/膨胀后的连通性问题，需要更细几何证据，最终类别 unresolved，不叫“目标找不到”或“深度传感器坏了”。

数据：[failure_evidence.json](../eagor_outputs/online_navigation_20260911/analysis/failure_evidence.json)。不把到达次数、reason 标签次数、是否检测到目标混成同一个指标。

## 8. 五方法公平比较与停止干预

同一地图、frontier、A*、执行配置；唯一方向条件差异。全部主实验使用**原 Area**。普通感知都是 Oracle Semantic；GT Direction 额外带诊断特权方向。

| 方向条件 | SR | SPL | 平均步数 | 平均路径 m | 平均阻塞碰撞 | 曾进入成功范围 |
|---|---:|---:|---:|---:|---:|---:|
| GT Direction | 0/10 | 0 | 256.5 | 27.72 | 0.6 | 5/10 |
| EAGOR | 0/10 | 0 | 240.6 | 26.10 | 0.7 | 4/10 |
| Grid | 0/10 | 0 | 230.3 | 23.95 | 0.8 | 4/10 |
| Centroid | 1/10 | 0.051 | 226.3 | 23.22 | 0.7 | 4/10 |
| 无方向探索 | 0/10 | 0 | 252.0 | 25.68 | 1.3 | 2/10 |

**【组会图 3 插入位置：统一 Area 停止的主结果，不与 Oracle SR 混画】**

![公平对比](../eagor_outputs/online_navigation_20260911/analysis/fair_comparison.png)

EAGOR、Grid 的“曾进入范围”都高于无方向探索，但样本少、轨迹不同、停止仍失败，**这最多说明此设置下方向引导存在接近目标的迹象，不证明 EAGOR 独有增益或统计显著优势**。Centroid 唯一成功是 Z6；不能据 1 个成功断言其普遍优于 EAGOR。没有额外加入 Circular 以扩展矩阵。

| 条件 | 有效方向样本 | 角 MAE | 感知+方向 ms | 建图 ms | 规划 ms | 跟随 ms |
|---|---:|---:|---:|---:|---:|---:|
| GT Direction | 2565 | 约 0° | 7.95 | 36.05 | 4.80 | 0.53 |
| EAGOR | 2406 | 41.26° | 22.14 | 56.11 | 6.66 | 0.94 |
| Grid | 2303 | 40.75° | 10.18 | 36.01 | 3.73 | 0.53 |
| Centroid | 2227 | 38.51° | 10.80 | 47.10 | 6.57 | 0.82 |
| 无方向探索 | 0 | N/A | 7.95 | 37.82 | 4.34 | 0.59 |

角误差只统计有效样本，对当前视点所有合法目标实例中心取最小角误差；不是固定实例跟踪精度。GT 方向的接近零误差是适配器一致性，不是感知能力。时间是逐任务均值再平均，方向时间包含感知；多个独立实验曾并行运行，CPU 竞争会影响数字，**不是严格串行硬件 benchmark**。VLM 调用均为 0，感知调用规则为每个新观测一次，候选多少不增加调用。

### 小规模停止干预：只补两个共同案例

选择依据事先来自主实验日志：oLB 和 pLe 都是 EAGOR、Grid 曾进入范围却最终失败的共同任务。不是全数据集 Oracle Stop 主结果，也不是无偏子样本。

| 任务 | EAGOR Area → Oracle Stop | Grid Area → Oracle Stop | Oracle Stop 时步数 / 路径 |
|---|---|---|---|
| oLBMNvg9in8_0 | 失败 → 成功 | 失败 → 成功 | 两者均 80 步 / 7.00 m |
| pLe4wQe7qrG_0 | 失败 → 成功 | 失败 → 成功 | 两者均 86 步 / 8.51 m |

4 次干预均为 0 阻塞碰撞，停止前动作/位姿前缀与各自主实验一致。[paired_stop_and_replay_audit.json](../eagor_outputs/online_navigation_20260911/analysis/paired_stop_and_replay_audit.json) 保存逐项断言。这里足以定位这些案例的漏停，但**没有解决非 Oracle stop**，也没有把事后最短距离喂回策略。

## 9. 实际视频与插图位置

视频为 1280×720、10 fps、每决策步一帧；是 8–30 秒的播放时长，不代表机器人只运动了这些秒。严格保留原 300 步预算，未为了视频加长 benchmark 或复制帧冒充更长任务。

左侧在线地图：unknown 深灰、free 浅灰、occupied 黑、膨胀带灰紫；绿色实际轨迹、蓝色规划路径、粉色子目标、金色候选。右侧当前 perspective RGB、ERP RGB 和诊断条件。没有叠加 GT 地图。

### 【视频 1 插入位置：真实门洞与跨房间成功】

[播放 2az GT 诊断视频](../eagor_outputs/online_navigation_20260911/case_videos_final/oracle_direction/2azQ1b91cZZ_0/navigation.mp4)：126 步、20 m、0 碰撞、成功。约第 53–89 帧（5.3–8.9 秒）可看出从大厅通过门洞进入相邻房间；后续继续绕过家具接近 cabinet。GT Direction 和 Oracle Stop 都在画面中明示。

![2az 实际抽帧](../eagor_outputs/online_navigation_20260911/analysis/case_videos_final_oracle_direction_2azQ1b91cZZ_0_frames.jpg)

### 【视频 2 插入位置：有隔断/立柱的绕行成功】

[播放 pLe GT 诊断视频](../eagor_outputs/online_navigation_20260911/case_videos_final/oracle_direction/pLe4wQe7qrG_0/navigation.mp4)：86 步、8.51 m、0 碰撞、成功。可见沿长条结构绕至侧边开口/相邻空间。`shower` 是数据集任务类别，不额外声称视觉中某一立柱就是语义目标。

[播放相同任务的 EAGOR + Oracle Stop 视频](../eagor_outputs/online_navigation_20260911/stop_diagnostic_final/eagor/pLe4wQe7qrG_0/navigation.mp4)。此视频不代表 EAGOR + Area 成功。

### 【视频 3 插入位置：代表性真实失败，不能删掉】

[播放 EU6 失败视频](../eagor_outputs/online_navigation_20260911/case_videos_final/oracle_direction/EU6Fwq7SyZv_0/navigation.mp4)：300 步、15.50 m、0 阻塞碰撞，预算耗尽。约 8.5 秒后可反复看到局部观察、候选变化和回退，说明“低碰撞”不等于有全局任务进度。

![EU6 实际失败抽帧](../eagor_outputs/online_navigation_20260911/analysis/case_videos_final_oracle_direction_EU6Fwq7SyZv_0_frames.jpg)

补充：[EAGOR oLB 成功停止诊断](../eagor_outputs/online_navigation_20260911/stop_diagnostic_final/eagor/oLBMNvg9in8_0/navigation.mp4)。共 7 个最终视频已检查能解码、帧数与日志一致；对应 3 个 GT 案例重跑和 4 个停止干预。旧 `case_videos` / `stop_diagnostic_subset` 保留早期渲染版本（路径配色与图例不符），应使用 `_final` 目录；重新渲染没有改变决策核心或成功结果。

## 10. 日志、可复现性与准确命令

每个 episode 输出 `task.json / summary.json / steps.json / steps.csv / frontier_scores.json`，以及每 25 步的 `map_NNN.npz / depth_NNN.npz`、最终地图、可选 `navigation.mp4`。depth 文件保留原 face-axial、转换后 radial、位姿与旋转；地图保留 origin、resolution、证据、观测状态、更新时间、visits、contacts、trajectory、safe/free/occupied。普通无视频运行不保存逐帧 RGB。

首次事件分别记为 `first_perception_detection`、`first_evaluation_instance_visible`、`first_success_region`。第二项定义为当前 ERP 至少 1 像素属于 episode 的 object_id，是新增评估可见性标注；官方 ObjectNav 没有同名额外可见成功条件。Oracle 感知类别命中与合法 episode 实例可见仍是不同事件，不应把前者自动当作真值发现。

失败 summary 只自动给出 `stop / budget_exhausted / unresolved` 等终止分类，更精确环节须结合证据分析。`analysis/aggregate.csv`、`episodes.json`、`failure_evidence.json` 保留完整数字，图表不能代替 raw log。

每个 MP3D 运行目录保存准确 `provenance.json.argv`、`config.yaml`、`habitat_config.yaml`、manifest、全 EAGOR/config 源码快照、逐文件 SHA256、git HEAD、工作区 diff、git status。由于原目录有大量用户未跟踪文件，单看 git diff 不够，源码快照才是重要补充。未 reset、提交、推送、覆盖旧实验。

以下全部从仓库根目录执行；`--output` 必须是**尚不存在的新目录**，示例后缀 `rerun01` 可自行换新名。脚本遇到已有运行目录会拒绝覆盖。

```bash
cd /home/xiaotian/navigation/habitat-lab

# 回归、编译、数据检查
conda run -n habitat python -m eagor_repro.scripts.run_tests
conda run -n habitat python -m compileall -q eagor_repro
conda run -n habitat python -m eagor_repro.scripts.evaluate_eagor --overlay configs/eagor/mp3d_oracle.yaml --preflight

# 实际 native ERP 深度几何校验；使用新目录，保留所有既有校验产物
conda run -n habitat python -m eagor_repro.scripts.verify_erp_depth_geometry --output eagor_outputs/online_geometry_rerun01/verified.json

# 四个闭环人工布局（不进入 MP3D 主结果）
conda run -n habitat python -m eagor_repro.scripts.verify_online_layouts --output eagor_outputs/online_layouts_rerun01

# 原同批 10 任务：隔离规划/执行的诊断
conda run --no-capture-output -n habitat python -m eagor_repro.scripts.run_online_navigation --output eagor_outputs/online_diag_rerun01 --methods oracle_direction --stop oracle_distance

# 五方法统一原 Area 主比较
conda run --no-capture-output -n habitat python -m eagor_repro.scripts.run_online_navigation --output eagor_outputs/online_fair_rerun01 --methods oracle_direction eagor grid centroid exploration --stop area

# 单任务：保持 Area，不误写为成功诊断
conda run --no-capture-output -n habitat python -m eagor_repro.scripts.run_online_navigation --output eagor_outputs/online_single_rerun01 --methods eagor --stop area --episodes pLe4wQe7qrG_0 --video

# 两个共同案例，只做停止干预诊断，并生成真实视频
conda run --no-capture-output -n habitat python -m eagor_repro.scripts.run_online_navigation --output eagor_outputs/online_stop_rerun01 --methods eagor grid --stop oracle_distance --episodes oLBMNvg9in8_0 pLe4wQe7qrG_0 --video

# GT 诊断的成功、门洞、失败视频
conda run --no-capture-output -n habitat python -m eagor_repro.scripts.run_online_navigation --output eagor_outputs/online_video_rerun01 --methods oracle_direction --stop oracle_distance --episodes 2azQ1b91cZZ_0 pLe4wQe7qrG_0 EU6Fwq7SyZv_0 --video

# 本次既有目录的离线汇总/配对审计；仅重建 analysis 派生产物，不重写 raw log
conda run -n habitat python -m eagor_repro.scripts.summarize_online_navigation --root eagor_outputs/online_navigation_20260911
conda run -n habitat python -m eagor_repro.scripts.audit_online_artifacts --root eagor_outputs/online_navigation_20260911
```

默认 manifest 路径在 `configs/eagor/mp3d_online.yaml`，迁移机器时应复制 manifest 与数据并显式覆盖路径。恢复命令是 `python -m eagor_repro.scripts.prepare_online_manifest --output <新的文件路径>`，依赖原 CSV 和原 MP3D dataset；不是随机重采样替代方案。

## 11. 可以支持与不能支持的论文表述

可以写：在固定 10 个 MP3D ObjectNav 任务、GT 位姿和受控方向/停止诊断下，新增无先验地图的在线深度建图及 frontier/A* 执行链路，将 GT Direct + Oracle Stop 的 0/10 提高到 6/10，并显著减少按原定义统计的完全阻塞碰撞；仍存在单层适用范围、局部探索停滞与停止判据问题。

不能写：完整复现论文导航性能、EAGOR 非 Oracle SR=60%、EAGOR 优于所有二维表示、完全无特权 RGB-D、无地图跨楼层能力、真实 VLM/真实机器人部署成功。新 online planner 是本项目新增工程变量，不冒称论文原生 SH-BF 的组成部分；原论文数学实现仍沿用既有版本，本轮没有重新审计整篇论文或运行其完整实验协议。

下一步的优先级是非 Oracle 停止和单层地图/探索鲁棒性：需要独立校验 stop 与合法 viewpoint 的差别，进一步检查 x8 连通性和 EU6 回退循环；之后才适合扩大真实感知实验。楼层拓扑、楼梯识别、噪声定位、强化学习、新 VLM 训练、真实机器狗部署都属于后续工作，本次没有实现。
