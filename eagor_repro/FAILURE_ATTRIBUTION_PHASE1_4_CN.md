# MP3D ObjectNav Failure Attribution：Phase 1–4 阶段报告

> **历史结果提示（2026-09-10）：** 本文的 A–H 仍可作为 3-scene 初始诊断记录；旧 row-I 只把 navmesh 折点转换为 heading 后交给普通 fixed-step controller，不是严格的离散动作最短路径上界。修正实现和 10-scene 结果（I=10/10、SPL=0.978、0 碰撞）见 [`FAILURE_ATTRIBUTION_10SCENES_CN.md`](FAILURE_ATTRIBUTION_10SCENES_CN.md)。论文和组会不应再引用本文旧 I=0.667 的数字或视频。

> 日期：2026-08-17  
> 范围：Oracle Direction、Oracle/Depth Stop、Depth Traversability Planner、Failure Attribution Matrix  
> 结论边界：3 个 MP3D unique scenes；所有行仍使用 Oracle semantic，因此都是导航诊断而非 zero-shot 结果  
> 总汇总：[failure_attribution_summary.md](../eagor_outputs/failure_attribution/summaries/failure_attribution_summary.md)

## 1. 最终回答

### 1.1 如果直接提供 GT target direction，当前 controller 是否仍然失败？

**是。** Case C 使用 GT direction、原始 direct controller 和 area stop，三个场景的 direction MAE 全部为 0°，但 SR 仍为 0，平均碰撞 25 次，主失败模式全部是 `blocked_path`。

这说明当前 SR=0 不能主要归因于 SH-BF direction estimation；即使方向完全正确，朝目标直走仍会撞向墙体、家具或不可达的对象方向。

### 1.2 direction 完全正确并加入 traversability awareness 后能否成功？

- 单帧 depth candidate planner（Case E）仍为 SR=0：局部深度只能判断当前视野中的 clearance，不能解决门、房间、拐角和 navmesh 拓扑。
- Habitat navmesh shortest-path next heading（Case I，明确的 Oracle upper bound）达到 **SR=0.667、SPL=0.659**：两个场景成功，一个场景仍因固定步长跟踪和碰撞恢复失败。

因此，真正缺失的是**全局/拓扑 path reachability**。固定步长 controller 在收到有效路径折点后可以完成部分 episode，但仍是次要瓶颈。

### 1.3 pixel-area stop 是否是主要失败来源之一？

它是**明确存在的贡献因素，但不是这三个场景的首要瓶颈**：

- Grid + Area 在 3 个场景中出现 1 次 false stop；
- GT Direction 下把 Area 换成 Oracle Distance（C→D），SR 仍是 0，因为机器人根本没有走到成功区域；
- Area + Depth 仍会 false stop：`2azQ1b91cZZ` 第 1 步 cabinet mask 的 median depth 为 0.614 m，但 Habitat 到有效 goal viewpoint 的距离为 9.178 m。

这说明“同类别语义像素很近”不等于“当前 episode 已进入成功区域”。Depth stop 还需要实例一致性、可达 viewpoint 或与任务 success definition 对齐的距离模型。

### 1.4 EAGOR belief 加入 planner 后是否转化为导航收益？

当前不能得出正收益结论。Case G（EAGOR + local depth planner + depth stop）仍为 SR=0。由于 Case E 的 Oracle direction 在同一个 local planner 下也为 SR=0，planner 本身尚未达到能够隔离 belief 差异的能力。

只有先提供可用的非 Oracle global/topological planner，再比较 EAGOR/Grid/Oracle direction 的 matched planner + matched stop，才能判断 EAGOR direction belief 是否转化为 SR/SPL 收益。因此本阶段**不进入 SH-BF Top-K 或 correlation-aware 修改**。

## 2. 瓶颈归因

在用户给出的选项中，当前证据支持：

> **主要瓶颈：C. Path Planning。次要瓶颈：B. Controller 和 D. Stop Criterion。A. Belief 暂未被证明是主瓶颈。**

核心对照：

```text
C: Oracle direction + Direct + Area           SR = 0.000
D: Oracle direction + Direct + Oracle stop    SR = 0.000
E: Oracle direction + Local depth + Oracle    SR = 0.000
I: Oracle direction + Navmesh path + Oracle   SR = 0.667
```

C→D 没有改善，排除了“仅修 stop 就能解决”；D→E 没有改善，说明单帧局部 depth 不等于可达路径；E→I 从 0 提升到 0.667，直接定位到 global/navmesh reachability。

## 3. 实验矩阵结果

| ID | Direction | Planner | Stop | SR ↑ | SPL ↑ | Steps ↓ | Path m | Collision ↓ | False Stop ↓ | MAE ↓ | Recovery |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | EAGOR | Direct | Area | 0.000 | 0.000 | 300.0 | 3.308 | 28.33 | 0.00 | 52.15° | 0.00 |
| B | Grid | Direct | Area | 0.000 | 0.000 | 208.0 | 1.709 | 16.33 | 0.33 | 38.43° | 0.00 |
| C | Oracle | Direct | Area | 0.000 | 0.000 | 300.0 | 0.801 | 25.00 | 0.00 | **0.00°** | 0.00 |
| D | Oracle | Direct | Oracle Distance | 0.000 | 0.000 | 300.0 | 0.801 | 25.00 | 0.00 | **0.00°** | 0.00 |
| E | Oracle | Depth Local | Oracle Distance | 0.000 | 0.000 | 300.0 | 0.903 | 25.00 | 0.00 | **0.00°** | 0.00 |
| F | Oracle | Depth Local | Area + Depth | 0.000 | 0.000 | 200.3 | 0.739 | 16.67 | 0.33 | **0.00°** | 0.00 |
| G | EAGOR | Depth Local | Area + Depth | 0.000 | 0.000 | 200.3 | 3.234 | 20.00 | 0.33 | 51.33° | 0.00 |
| H | Grid | Depth Local | Area + Depth | 0.000 | 0.000 | 115.3 | 2.259 | 8.67 | 0.67 | 46.45° | 0.00 |
| I | Oracle | Oracle Navmesh | Oracle Distance | **0.667** | **0.659** | 157.7 | 10.707 | 8.33 | 0.00 | **0.00°** | 0.00 |

Case I 是在要求的 A–H 之外增加的 path-planning upper bound。它真实调用 Habitat navmesh `MultiGoalShortestPath`，没有伪造 oracle path。

<!-- ===== 图 1 插入位置：Phase 1–4 failure attribution ===== -->

![Phase 1–4 failure attribution](../eagor_outputs/failure_attribution/summaries/failure_attribution.png)

建议图注：*三个 MP3D 场景上的模块化导航归因。GT direction 和 Oracle stop 无法挽救 direct/local-depth 导航；真实 navmesh path upper bound 将 SR 提高到 0.667，表明 path reachability 是当前最强瓶颈。所有实验使用 Oracle semantic。*

## 4. Stop 对比

| Stop Method | 对应 Case | Direction | Planner | False Stop ↓ | Correct Stop ↑ | SR ↑ |
|---|---|---|---|---:|---:|---:|
| Area | A | EAGOR | Direct | 0.000 | 0.000 | 0.000 |
| Area | B | Grid | Direct | 0.333 | 0.000 | 0.000 |
| Area | C | Oracle | Direct | 0.000 | 0.000 | 0.000 |
| Area + Depth | F | Oracle | Depth Local | 0.333 | 0.000 | 0.000 |
| Oracle Distance | D | Oracle | Direct | 0.000 | 0.000 | 0.000 |

C 与 D 是 stop 的干净隔离；F 同时包含 depth planner，因此只用于展示 practical depth stop 仍会发生 false stop。

## 5. Oracle navmesh 成功视频

<!-- ===== 图 2 插入位置：Oracle navmesh success ===== -->

![Oracle navmesh success frame](../eagor_outputs/failure_attribution/summaries/oracle_nav_success_frame.png)

- [▶ `2azQ1b91cZZ`：48 步成功，9.25 m，0 碰撞](../eagor_outputs/failure_attribution/I_oracle_navmesh_oracle_stop/oracle_direction/videos/episode_2azQ1b91cZZ_0.mp4)
- [▶ `8194nk5LbLH`：125 步成功，17.33 m，3 碰撞](../eagor_outputs/failure_attribution/I_oracle_navmesh_oracle_stop/oracle_direction/videos/episode_8194nk5LbLH_0.mp4)
- [▶ `EU6Fwq7SyZv`：300 步失败，5.54 m，22 碰撞](../eagor_outputs/failure_attribution/I_oracle_navmesh_oracle_stop/oracle_direction/videos/episode_EU6Fwq7SyZv_0.mp4)

视频 dashboard 标出 belief target heading、planner selected heading、target depth、stop mode、planner status 和在线 failure signal。GT 进入控制的 case 会显示 `ORACLE UPPER BOUND`。

## 6. Local depth planner 的补充敏感性检查

主矩阵使用 25th-percentile sector clearance。根据碰撞日志又测试了更保守的 percentile：

| Clearance percentile | SR | Mean collisions | Mean planner recoveries | 解释 |
|---:|---:|---:|---:|---|
| 25（主矩阵） | 0.000 | 25.00 | 0.00 | 对机器人侧缘/navmesh 阻塞不敏感 |
| 10 | 0.000 | 26.33 | 1.33 | 没有改善 |
| 5 | 0.000 | 19.00 | 100.00 | 碰撞降低 24%，但大量“全方向阻塞”并原地恢复 |

5th-percentile 的碰撞下降不是有效导航收益，因为 SR 仍为 0 且恢复次数急剧增加。因此保留可解释的 25th-percentile 主设置，不用调参掩盖 local planner 的能力边界。

## 7. Failure attribution 规则

每个失败 episode 输出一个 primary failure 和多个 secondary failures。固定优先级为：

```text
false_stop
→ planner_no_feasible_heading
→ blocked_path
→ direction_error
→ oscillation
→ target_switch
→ lost_target
→ collision
→ timeout
```

优先级先放终止性/直接阻塞原因，再放方向误差和行为症状。它是确定性的诊断规则，不应被表述为学习得到的因果模型。

Episode summary 新增：

```text
primary_failure_mode
secondary_failure_modes
num_stop_attempts
correct_stop / false_stop / missed_stop
distance_at_stop_m
target_visible_at_stop
target_depth_at_stop_m
blocked_step_count
planner_recovery_count
oscillation_count
target_switch_count
lost_target_step_count
oracle_upper_bound
```

## 8. 代码修改清单

### Created

| 文件 | 作用 |
|---|---|
| [`policies/oracle_direction_policy.py`](policies/oracle_direction_policy.py) | 最近有效目标实例的 GT egocentric bearing；仅诊断 |
| [`controllers/stop_criteria.py`](controllers/stop_criteria.py) | `area` / `area_depth` / `oracle_distance` 独立 stop |
| [`planning/local_traversability.py`](planning/local_traversability.py) | ERP depth angular sector 的 robust clearance |
| [`planning/candidate_heading_planner.py`](planning/candidate_heading_planner.py) | Direct / Depth Local / Oracle Navmesh heading 接口 |
| [`planning/__init__.py`](planning/__init__.py) | planning package 导出 |
| [`evaluation/failure_attribution.py`](evaluation/failure_attribution.py) | primary/secondary failure 规则 |
| [`scripts/evaluate_failure_attribution.py`](scripts/evaluate_failure_attribution.py) | A–I 矩阵、CSV/MD/PNG、自动诊断和断点续跑 |
| [`tests/test_navigation_attribution.py`](tests/test_navigation_attribution.py) | Oracle direction、stop、planner、归因测试 |
| [`../configs/eagor/mp3d_failure_attribution.yaml`](../configs/eagor/mp3d_failure_attribution.yaml) | 独立 MP3D 归因配置，不覆盖旧结果 |

### Modified

| 文件 | 修改与预期效果 |
|---|---|
| [`config.py`](config.py) | 新 policy/planner/stop factory；所有模块由 YAML 开关 |
| [`sensors/panorama_sensor.py`](sensors/panorama_sensor.py) | 新增同视点 metric ERP depth，并保持原水平 convention |
| [`evaluation/habitat_evaluator.py`](evaluation/habitat_evaluator.py) | 串联 Policy → Planner → Stop → Controller；GT 分支明确标记 |
| [`evaluation/evaluator.py`](evaluation/evaluator.py) | 扩充逐步 CSV 字段 |
| [`evaluation/video_renderer.py`](evaluation/video_renderer.py) | 增加 planner/stop/depth/failure telemetry 和 Oracle 警示 |
| [`policies/__init__.py`](policies/__init__.py)、[`controllers/__init__.py`](controllers/__init__.py) | 导出新模块 |
| [`scripts/run_tests.py`](scripts/run_tests.py) | 纳入新测试模块 |
| [`../configs/eagor/base.yaml`](../configs/eagor/base.yaml) | 默认仍为 `direct + area`，新增可覆盖的 planning/stop 参数 |

### Intentionally unchanged

以下文件刻意未改，防止把导航失败混入球谐数学变化：

- [`spherical/spherical_grid.py`](spherical/spherical_grid.py)
- [`spherical/spherical_harmonics.py`](spherical/spherical_harmonics.py)
- [`spherical/rotation.py`](spherical/rotation.py)
- [`spherical/belief_filter.py`](spherical/belief_filter.py)
- [`controllers/fixed_step_controller.py`](controllers/fixed_step_controller.py)
- 原 Centroid / Circular Centroid / Grid / EAGOR policy 数学逻辑
- Oracle semantic 和 Qwen grounding 实现

## 9. 测试结果

```text
原有测试：35 / 35 PASS
新增测试：17 / 17 PASS
总计：    52 / 52 PASS
compileall: PASS
MP3D preflight: ready=true, scene_count=90
```

新增测试覆盖：

- target front / left / right；
- agent yaw 改变后的 egocentric Oracle direction；
- far / near / empty-mask / invalid / partial-invalid depth stop；
- Habitat strict success-distance Oracle stop；
- target free、left free、right free、all blocked、±π wrap；
- oracle_nav 必须显式获得 Habitat shortest-path heading；
- primary false-stop attribution 优先级。

## 10. 复现命令

```bash
cd /home/xiaotian/navigation/habitat-lab

conda run -n habitat python -m eagor_repro.scripts.run_tests

conda run -n habitat python -m eagor_repro.scripts.evaluate_failure_attribution \
  --config configs/eagor/base.yaml \
  --overlay configs/eagor/mp3d_failure_attribution.yaml
```

中断后复用已完成 case：

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_failure_attribution \
  --resume
```

只运行某些 case 或关闭视频：

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_failure_attribution \
  --experiments C D E I --no-video
```

## 11. 输出目录

```text
eagor_outputs/failure_attribution/
├── A_eagor_direct_area/
├── B_grid_direct_area/
├── C_oracle_direct_area/
├── D_oracle_direct_oracle_stop/
├── E_oracle_depth_planner_oracle_stop/
├── F_oracle_depth_planner_depth_stop/
├── G_eagor_depth_planner_depth_stop/
├── H_grid_depth_planner_depth_stop/
├── I_oracle_navmesh_oracle_stop/
└── summaries/
    ├── failure_attribution.csv
    ├── failure_attribution.md
    ├── failure_attribution_summary.md
    ├── failure_attribution.png
    ├── failure_cases.csv
    ├── stop_comparison.csv
    └── oracle_nav_success_frame.png
```

总计生成 27 个 case/scene dashboard MP4，不覆盖之前的 `mp3d_objectnav_multiscene` 结果。

## 12. 下一步

1. 不改 SH-BF，先实现非 Oracle 的 map/topological planner：depth occupancy、frontier/doorway exploration、局部避障与全局重规划。
2. 用同一 planner + Oracle stop 比较 Oracle/EAGOR/Grid direction，先隔离 belief 对路径目标选择的影响。
3. 让 depth stop 绑定对象实例与可达 viewpoint，而不是只看 category mask median depth。
4. 修复 Case I 的 `EU6Fwq7SyZv` waypoint-following：减少折点附近振荡，并记录 path-progress/stuck 指标。
5. 只有当可用 planner 下 Oracle 与 EAGOR 仍有显著差距时，才进入 Phase 5 Top-K peaks；当前不进入 correlation-aware 或 translation-aware belief。
