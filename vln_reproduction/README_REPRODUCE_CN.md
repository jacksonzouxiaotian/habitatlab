# VLN 基线复现与迁移索引

更新：2026-09-18。此文件保留已有工作和后续复现入口；**入口存在、下载完成、接口 smoke 都不等于论文完整复现**。EAGOR 是另一条 ObjectNav/方向诊断工作线，不混入 VLN 排名。

## 1. 必须保留的基线

| 方法 | 代码与运行入口 | 本地状态与注意事项 |
|---|---|---|
| NaVILA | 外部 `NaVILA/evaluation/run.py`，本仓库 `examples/narrow_passage_rl/vln_r2r_preflight.py` | 已有运行产物保存在外盘 `vln/formal_runs/`；当前 trainer 使用 4-bit，属于本地量化配置，不能冒称官方原精度结果 |
| NaVILA + DEGNAV | 同上，`EVAL.DEGNAV_ADAPTER.ENABLED True` | 额外依赖本仓库 `vln_depth_safety.py`；必须恢复 NaVILA 补丁、相同宏动作及碰撞记忆行为 |
| Open-Nav | 外部 `Open-Nav/`、其固定版本 README；本仓库 `opennav_preflight.py` | 已有资产/环境预检，不代表正式评估完成；还依赖 SpatialBot、recognize-anything、waypoint predictor 等 |
| VLN-CE | 外部 `VLN-CE/` 的 README 与配置 | 代码/基础环境保留，不能仅凭克隆代码认定其各学习基线已训练或评估 |
| NaVid-current-public | 外部 `NaVid-VLN-CE/`；`fetch_assets.py navid`；接口 smoke | 权重下载与依赖安装已完成，完整模型推理和正式 SR/SPL 未验收；不等同 RSS 2024 原始 checkpoint |
| Uni-NaVid | 外部 `Uni-NaVid/`；`fetch_assets.py uninavid` | 保留源码、样例与审计，完整模型推理/正式评估未完成；VLN/ObjectNav/EQA/Following 必须分任务验收 |
| AwareVLN | 外部 `AwareVLN/`；`fetch_assets.py awarevln` | 保留源码、配置与审计，完整模型未验收；最终 `awarevln/` 权重不能用初始化 NaVILA 权重替代 |

NaVid/Uni-NaVid 现有 evaluator 的 GT 距离停止、AwareVLN 的恢复/无效输出处理仍须按 `paper_reproduction_matrix.csv` 审计隔离。不能把修正停止协议后的结果与未修正的历史结果直接混表。该 CSV 与 `REPRO_STATE.md` 是前期阶段快照：其中“NaVid 下载未完成”已被后续安装/下载日志更新，但没有新增模型导航成功结论。

## 2. Git 保留什么、不保留什么

- 本仓库代码、配置、测试、说明及迁移工具。
- `migration/external_sources/manifest.json`：10 个外部源码仓库的 URL、精确 commit 和原路径，包含旧 Habitat-Lab/Sim、Open-Nav 两个附属仓库。
- `migration/external_sources/asset_revisions.json`：已审计的 NaVid、Uni-NaVid、AwareVLN 模型/样例仓库 revision，不包含权重。
- `migration/external_sources/patches/NaVILA.patch`：量化兼容、安全适配器、非学习基线 episode 数处理等本地改动。
- `migration/external_sources/patches/NaVid-VLN-CE.patch`：NumPy 兼容和动作解析修复；无效输出不再随机移动。
- **不上传** MP3D/HM3D/R2R 原始数据、模型权重、Conda 环境、视频、大体积原始日志或访问凭据。现有历史提交可能已有较小结果图表；本次不重写历史。

完整状态应按 [迁移手册](../migration/README_MIGRATION_CN.md) 搬迁六个目录和四套环境。Git 不能替代它：外部仓库未跟踪资产、site-packages 兼容改动、场景、模型和原任务 manifest 都仍需迁移。EAGOR 的原始 `eagor_outputs/online_navigation_20260911/episode_manifest.json` 也在本地数据迁移范围，不在本次新增代码上传范围。

若只恢复外部代码，使用**全新目录**：

```bash
python3 migration/restore_external_sources.py --root /data/vln_pinned_sources
# 检查计划后：克隆、checkout 固定 commit、恢复 submodules、校验并应用补丁
python3 migration/restore_external_sources.py --root /data/vln_pinned_sources --apply
```

该命令不会安装依赖或下载模型；输出分为 `vla/` 与 `vln_sources/`。按迁移手册恢复原路径兼容链接或逐项指定实际目录。若已有完整源目录副本，**不要再次应用补丁**。每个仓库的第三方许可证继续有效；这里保留版本和差异，不改变其许可。

## 3. 新机验收顺序

目标 Ubuntu 22.04、NVIDIA GPU；具体 GPU/驱动仍须实测。不要将 Habitat 0.3.3 与旧 VLN 的 0.1.7 混装。

1. 按迁移手册恢复路径、四环境并执行 `new_machine.sh check`。
2. 使用新输出目录做资产预检和接口测试。
3. NaVILA 先做 1 episode 模型评估，再按原 10 shards 配置正式评估；适配器条件独立输出。
4. NaVid → Uni-NaVid → AwareVLN：先模型真实推理，再固定任务 1/10/100/full；未运行项不得填写 0 分。
5. Open-Nav/VLN-CE：先核验固定源码中的数据和权重要求，独立做模型/episode smoke；不沿用 NaVILA 结果。

### NaVILA 与 NaVILA + DEGNAV

精确的模型路径、命令、10-shard 方式在 [迁移手册第 7.6 节](../migration/README_MIGRATION_CN.md#76-navila--vln-ce)。保持 4-bit，原始条件和适配器条件分别运行；本仓库必须加入 `PYTHONPATH`。

### Open-Nav 预检

从仓库根目录执行（替换 Python 路径）：

```bash
/data/habitat_migration/envs/navila-eval/bin/python \
  examples/narrow_passage_rl/opennav_preflight.py \
  --opennav-root /home/xiaotian/vla/Open-Nav \
  --output-dir migration_runs/opennav_preflight
```

这是预检而非导航评估。读取输出的缺失资产列表及固定版本 Open-Nav README 后再配置实验；不要自动调用有费用的外部服务。

### 三个新方法

```bash
/data/habitat_migration/envs/navila-eval/bin/python \
  -m unittest discover -s vln_reproduction -p 'test_*.py' -v
/data/habitat_migration/envs/navila-eval/bin/python \
  vln_reproduction/simulator_smoke.py --method navid
/data/habitat_migration/envs/navila-eval/bin/python \
  vln_reproduction/simulator_smoke.py --method uninavid
```

上述视频只有固定动作接口检查；不是模型导航。正式模型推理使用各自匹配环境，不能因为接口测试在 navila-eval 通过就认定 NaVid 训练环境成立。

若缺模型，`fetch_assets.py` 使用 Hugging Face 元数据 revision 固定下载；先迁移 `vln/reproduction/assets/<method>/revision.json`，否则首次运行会重新选取当时的版本。使用已安装 `huggingface_hub` 的环境，例如 navid：

```bash
/data/habitat_migration/envs/navid/bin/python vln_reproduction/fetch_assets.py navid --download
# 根据资源和许可分别选择，不要一次盲目下载全部模型：
# ... fetch_assets.py uninavid --download
# ... fetch_assets.py awarevln --download
```

`vln/reproduction/attempts/*/result.json`、`assets/*/download_verified.json`、`audits/` 和训练复现 CSV 都要随完整迁移保留。`audit_local.py` 还依赖旧下载文件夹中的原请求文档；若该文件未搬迁，优先读取已迁移 audit，不盲目重跑。正式实测用 `run_logged.py --label <新名称> --cwd <源码目录> -- <实际命令>` 保存退出码及输出；没有已验证的完整模型评估命令时，不把官方模板说成已成功运行。

## 4. 对照与披露

保持任务集合、seed、相机、动作尺度、预算、停止协议和模型精度一致；首次到达效率只在共同到达子集比较。训练复现与评估复现分开。GT pose/Oracle Semantic/EAGOR 的方向/停止诊断不能替代 RGB+语言 VLN；离线样例、接口 smoke、模型推理、正式仿真、导航训练及真实机器人分别验收。
