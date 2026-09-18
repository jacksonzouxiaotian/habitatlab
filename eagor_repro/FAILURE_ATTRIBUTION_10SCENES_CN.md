# EAGOR × MP3D：10 场景导航归因复现实验

> 实验日期：2026-09-10  
> 状态：已完成并复核  
> 评测性质：Oracle-semantic 导航诊断，不是论文正式 zero-shot benchmark  
> 机器可读结果：[CSV](../eagor_outputs/failure_attribution_10scenes_corrected_20260910/summaries/failure_attribution.csv)；[逐 episode failure cases](../eagor_outputs/failure_attribution_10scenes_corrected_20260910/summaries/failure_cases.csv)

## 1. 本轮结论

在 10 个不同 MP3D 场景、每条最长 300 步的同场景对照中：

- EAGOR、Grid、GT direction 只要仍采用“朝目标方向直走”，成功率均为 **0/10**；
- GT direction 的 MAE 已是 **0°**，仍然 0/10，因此当前失败不能归因于 coefficient 更新或方向解码；
- 把 area stop 换成 Oracle distance stop，结果仍为 0/10；
- 加入单帧 ERP depth candidate-heading planner，仍为 0/10；
- 用 Habitat 的 navmesh 最短路径和离散动作 follower 后达到 **10/10，SR=1.000，SPL=0.978，0 碰撞**。

因此当前最强、且已被干预实验隔离出的瓶颈是：

> **缺少非 Oracle 的全局/拓扑可达性规划。当前不应优先修改 SH-BF coefficient update。**

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| 数据 | MP3D ObjectNav v1 val episodes |
| 场景采样 | episode 原始顺序中每个场景取第一条，共 10 个 unique scenes |
| 最大步数 | 300 |
| 动作空间 | `stop`、`move_forward`、`turn_left`、`turn_right` |
| 前进一步 | 0.25 m |
| 每次转向 | 15° |
| 观测 | 原生 512×256 ERP RGB、depth、semantic |
| 感知 | Oracle semantic，用来隔离导航，不计作 zero-shot |
| SH-BF | 论文式 coefficient update；本轮未修改 |
| 成功半径 | 0.1 m，与本地 Habitat task 配置一致 |
| 随机种子 | 0 |

10 个场景及目标类别为：`Z6MFQCViBuw/chest_of_drawers`、`EU6Fwq7SyZv/cabinet`、`2azQ1b91cZZ/cabinet`、`X7HyMhZNoso/bed`、`oLBMNvg9in8/cabinet`、`8194nk5LbLH/gym_equipment`、`TbHJrupSAjP/counter`、`pLe4wQe7qrG/shower`、`x8F5xyUWy9e/plant`、`QUCTc6BB5sX/chest_of_drawers`。

## 3. 对照矩阵

| ID | Direction | Planner | Stop | SR ↑ | SPL ↑ | 平均步数 ↓ | 平均路径 m | 平均碰撞 ↓ | False stop ↓ | MAE ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | EAGOR | Direct | Area | 0.000 | 0.000 | 300.0 | 2.290 | 29.2 | 0.0 | 37.40° |
| B | Grid | Direct | Area | 0.000 | 0.000 | 244.2 | 2.432 | 19.4 | 0.2 | 34.84° |
| C | Oracle | Direct | Area | 0.000 | 0.000 | 300.0 | 1.524 | 24.6 | 0.0 | **0.00°** |
| D | Oracle | Direct | Oracle distance | 0.000 | 0.000 | 300.0 | 1.524 | 24.6 | 0.0 | **0.00°** |
| E | Oracle | Depth local | Oracle distance | 0.000 | 0.000 | 300.0 | 1.581 | 24.7 | 0.0 | **0.00°** |
| I | Oracle | Habitat navmesh follower | Oracle distance | **1.000** | **0.978** | **74.8** | 11.908 | **0.0** | **0.0** | **0.00°** |

这里的 I 是诊断上界：它使用 episode goal viewpoints 和完整 navmesh，不能作为待投稿方法的结果。它的用途是回答“这些 episode 是否可达，以及当前 controller 之外是否存在有效路径”。

<!-- ===== 组会图 1 / 论文诊断图插入位置 ===== -->

![10 场景 failure attribution](../eagor_outputs/failure_attribution_10scenes_corrected_20260910/summaries/failure_attribution.png)

建议图注：*10 个 MP3D 场景的模块化导航归因。即使目标方向和停止距离完全正确，Direct 与单帧 Depth Local 仍为 0/10；Habitat navmesh 离散最短路径 follower 达到 10/10，说明 global/path reachability 是当前主瓶颈。所有结果均使用 Oracle semantic。*

## 4. 因果式对照解读

### 4.1 Direction：A/B → C

C 的方向误差从 A/B 的 37.40°/34.84°降为 0°，成功率仍为 0。GT bearing 只指出对象相对方向，直线方向可能穿过墙、家具或不同房间。因此本阶段没有证据支持继续调 SH 阶数、Top-K decode 或 coefficient 权重能改善 SR。

### 4.2 Stop：C → D

C 与 D 的轨迹、碰撞和成功率相同。Oracle stop 没有机会发挥作用，因为机器人没有进入成功区域。这排除了“只修停止阈值即可解决当前失败”的解释。

Grid 的 10 条轨迹中有 2 条 area false stop，说明 stop weakness 确实存在，但它是次要问题，不是当前 0 SR 的充分解释。

### 4.3 单帧局部避障：D → E

E 的 SR 仍为 0，平均碰撞还从 24.6 变为 24.7。单帧 ERP depth 能估计当前方向的局部 clearance，但没有跨帧地图、已访问状态、frontier、门洞拓扑或长程 waypoint，无法决定应当从墙体哪一侧绕行。

### 4.4 全局可达路径：E → I

唯一改变为提供 Habitat navmesh 可达路径及其离散动作 follower，SR 从 0 提升到 1.0，碰撞从 24.7 降到 0。这是本轮最强的归因证据。

## 5. 代表性 failure 与 success 可视化

### 5.1 同一场景的 Depth Local failure

`8194nk5LbLH` 中，GT direction 的 MAE=0°，但局部 planner 最终进入碰撞恢复循环，300 步仍未到达。

<!-- ===== 组会图 2 插入位置：局部规划失败帧 ===== -->

![Oracle direction + depth local failure](../eagor_outputs/failure_attribution_10scenes_corrected_20260910/summaries/oracle_depth_local_failure_frame.png)

[▶ 查看完整 300 步失败视频](../eagor_outputs/failure_attribution/E_oracle_depth_planner_oracle_stop/oracle_direction/videos/episode_8194nk5LbLH_0.mp4)

### 5.2 同一场景的 navmesh success

修正后的 I 沿全局路径绕开障碍，在 100 步、17.50 m 后成功，SPL=0.976，碰撞为 0。

<!-- ===== 组会图 3 插入位置：navmesh 成功帧 ===== -->

![Corrected Habitat navmesh follower success](../eagor_outputs/oracle_nav_follower_videos_3_20260910/summaries/oracle_navmesh_corrected_frame.png)

- [▶ `8194nk5LbLH`：100 步，17.50 m，0 碰撞](../eagor_outputs/oracle_nav_follower_videos_3_20260910/I_oracle_navmesh_oracle_stop/oracle_direction/videos/episode_8194nk5LbLH_0.mp4)
- [▶ `2azQ1b91cZZ`：55 步，9.25 m，0 碰撞](../eagor_outputs/oracle_nav_follower_videos_3_20260910/I_oracle_navmesh_oracle_stop/oracle_direction/videos/episode_2azQ1b91cZZ_0.mp4)
- [▶ `EU6Fwq7SyZv`：66 步，10.86 m，0 碰撞](../eagor_outputs/oracle_nav_follower_videos_3_20260910/I_oracle_navmesh_oracle_stop/oracle_direction/videos/episode_EU6Fwq7SyZv_0.mp4)

这些视频底部明确显示 `ORACLE UPPER BOUND`，不能与 EAGOR/Grid 的非 Oracle planner 结果混称。

## 6. 旧 I 实现为何作废

最初的 3 场景 I 得到 SR=0.667；本轮第一次扩到 10 场景时得到 SR=0.300。逐步日志显示它发生了固定模式的碰撞—转向—重新对准循环。

根因是旧实现只取 `MultiGoalShortestPath` 折线的下一个点，将它变成 heading 后仍交给普通 fixed-step controller。0.25 m 的离散步长可能越过很近的折点；首次碰撞又触发 6 步通用探索恢复。这不是严格的 action-space shortest-path upper bound。

修正后保留 `MultiGoalShortestPath` 选择最近有效 goal viewpoint，并把选中的终点交给 Habitat `ShortestPathFollower` / `GreedyGeodesicFollower` 产生离散动作。修正结果为：

| 检查 | SR | SPL | 平均碰撞 |
|---|---:|---:|---:|
| 修正后 3-scene smoke | 1.000 | 0.986 | 0.0 |
| 修正后 10-scene | 1.000 | 0.978 | 0.0 |

因此旧的 I=0.667/0.300 只保留为实现审计记录，不再作为论文或组会结论。A–E 未经过 oracle follower 分支，结果不受这次修正影响。

## 7. 与已发现两个 belief weakness 的关系

当前仓库已有两项独立、可重复生成的 SH-BF weakness：

1. **相关证据重复计数**：论文式加性更新会对高度相关的连续观测重复累积，使 coefficient 幅值和置信度随时间增长；[图与 CSV](../eagor_outputs/belief_weaknesses/)。
2. **平移视差未建模**：belief 只按旋转传播，机器人发生平移时，近目标真实 bearing 会变化，但旧 belief 不会按深度/平移校正；[图与 CSV](../eagor_outputs/belief_weaknesses/)。

本轮结果没有否定这两个数学 weakness，但给出了优先级：在现有 MP3D 闭环中，即使用 GT direction 完全绕过 belief，Direct/Depth Local 仍然失败。因此应先补齐非 Oracle reachability planner，再在 matched planner/stop 下测量这两个 weakness 对 SR/SPL 的真实影响。

## 8. 失败案例与当前 limitation

- A：10/10 primary failure 均为 `blocked_path`。
- B：8/10 为 `blocked_path`，2/10 为 `false_stop`。
- C/D/E：各 10/10 均为 `blocked_path`。
- I：10/10 成功，无 primary failure。

结果的边界必须写清楚：

- 只有 10 个 unique scenes，且每场景只取第一条 episode，不是完整 val 统计；
- 感知使用 Oracle semantic，没有评估真实 VLM 的漏检、误检和延迟；
- I 使用 GT goal viewpoints 与 navmesh，是诊断上界；
- 本轮没有 F/G/H，因此没有在同一可用的**非 Oracle** planner 下完成 Oracle/EAGOR/Grid direction 对照；
- 论文 v1 未给出所有实现和 benchmark 细节，当前不能宣称完整复现论文最终数值。

## 9. 下一阶段实验

下一步应实现一个不读取 episode goal/navmesh shortest path 的可达性模块，例如 depth 投影局部占据栅格 + pose 累积 + frontier/topological waypoint，并保持以下控制变量：

```text
same non-Oracle planner + same stop
├── Oracle direction（诊断上界）
├── EAGOR SH-BF
├── Grid belief
└── Circular centroid / Centroid
```

只有当 Oracle direction 在该 planner 上已经取得非零且稳定的 SR，EAGOR 与 Grid 的 SR/SPL 差异才可以归因到 belief。若 Oracle 仍失败，应继续修 planner/controller，而不是调整 coefficient update。

## 10. 复现命令

```bash
cd /home/xiaotian/navigation/habitat-lab

# 53 项回归测试
conda run -n habitat python -m eagor_repro.scripts.run_tests

# 10-scene A/B/C/D/E 导航诊断
conda run -n habitat python -m eagor_repro.scripts.evaluate_failure_attribution \
  --experiments A B C D E \
  --no-video \
  --override evaluation.num_episodes=10 \
  --override output.root=eagor_outputs/failure_attribution_10scenes_new_run

# 修正后的 10-scene navmesh action-space upper bound
conda run -n habitat python -m eagor_repro.scripts.evaluate_failure_attribution \
  --experiments I \
  --no-video \
  --override evaluation.num_episodes=10 \
  --override output.root=eagor_outputs/oracle_nav_follower_10scenes_new_run
```

运行中出现的 Conda `libtinfo.so.6`、Gym deprecation 和重复静态插件提示没有导致本轮失败；所有评测进程均以 exit code 0 结束。

## 11. 最终输出位置

```text
eagor_outputs/
├── failure_attribution_10scenes_corrected_20260910/
│   └── summaries/
│       ├── failure_attribution.csv
│       ├── failure_attribution.png
│       ├── failure_cases.csv
│       ├── oracle_depth_local_failure_frame.png
│       └── stop_comparison.csv
├── oracle_nav_follower_10scenes_20260910/
│   └── ... corrected 10-scene I raw CSV and JSON
└── oracle_nav_follower_videos_3_20260910/
    └── ... three corrected dashboard MP4s and success frame
```

