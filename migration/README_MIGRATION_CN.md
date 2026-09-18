# Habitat-Lab 全部相关工作：迁移与复现手册

核查日期：2026-09-18。这是**当前实际工作区**的迁移说明，不是重新 clone 官方 Habitat 的安装教程。

## 1. 先看结论

建议采用“**完整工作区 + 仓库外数据/模型/源码 + 原环境打包 + 路径兼容 + 分层验收**”。只复制 Git commit、只导出 requirements.txt，或只复制 `habitat-lab/data` 软链接，都不足以恢复当前工作。

本次已实际导出元数据快照：

```text
/media/xiaotian/ACD525D1B7A093D9/habitat_migration_snapshot_20260918/
```

内含四个环境的 Conda 完整导出、history、explicit 包列表、pip freeze/list/check；10 个源码仓库的 commit/status/diff/未跟踪文件清单；62 个源码树软链接记录；24 个关键配置、数据、模型文件的 SHA-256。**该目录只是元数据，不是已经打好的完整迁移包。** 本轮未复制约 150 GB 数据，未打包环境，也未在目标电脑验收。

默认目标为同架构 Linux x86_64、Ubuntu 22.04、NVIDIA GPU。其他系统/架构需单独重建原生依赖；不能将桌面 Linux 环境包直接用于 Windows、macOS 或 Jetson ARM。数据许可随数据保留，不上传到公开 GitHub。

用户已确认新机为 **Ubuntu 22.04 + NVIDIA GPU**；显卡型号、驱动、CPU 架构和目标目录仍需在新机确认。以下 `/data/habitat_migration` 是示例，不代表已经确认的新机路径。

## 2. 所有工作线、入口与完成边界

| 工作线 | 主要代码/记录 | 恢复所需环境 | 当前边界 |
|---|---|---|---|
| 程序化窄通道、几何 FSM、PPO/SAC/TD3、循环策略、BC/DAgger、失败记忆、可行性与消融 | `examples/narrow_passage_rl/`、`README_NARROW_PASSAGE.md`、`results/narrow_passage_rl/results_manifest.yaml` | habitat | 多套历史实验，必须按 manifest 区分正式、诊断和无效结果 |
| HM3D/Habitat 窄通道任务、传感器、几何与连续控制 | `habitat-lab/habitat/tasks/narrow_passage/`、对应 config、`data/datasets/narrow_passage/` | habitat | 本地 Habitat-Lab fork，不可替换为干净 upstream |
| 官方 MP3D PointNav 基线 | `eval_official_pointnav_baselines.py`、`results/official_pointnav_mp3d_v1_20260904/` | habitat | 正式 495 任务/种子；DD-PPO 是 Gibson-2+→MP3D 迁移；navmesh 方法是 Oracle |
| MP3D 派生窄通道、19D、DEGNAV-E2E | `collect_habitat_e2e_teacher.py`、`train_degnav_e2e_bc.py`、`eval_degnav_e2e.py`、`results/degnav_e2e_mp3d_narrow_v1_20260904/` | habitat | 400 train/80 val；E2E 是离线 BC 高层模式策略，不是 PPO；旧无效结果保留但不混用 |
| NaVILA、NaVILA+DEGNAV、VLN 离线诊断和 R2R | 仓库内 `vln_*`、`eval_vln_safety_adapter.py`；仓库外 `/home/xiaotian/vla/NaVILA/` 和 VLN `formal_runs/` | navila / navila-eval | 现有评估 trainer 显式 `load_4bit=True`，不能迁移后改为另一精度却称严格复现 |
| Open-Nav/VLN-CE 资产与环境预检 | `opennav_preflight.py`、外部 `Open-Nav/`、`VLN-CE/` | navila-eval | 资产预检不等于全部正式实验完成 |
| EAGOR 球谐 belief、方向基线、在线地图/Frontier/A*、停止归因 | `eagor_repro/`、`configs/eagor/`、`eagor_outputs/` | habitat | 原 10 任务 × 10 条件 = 100 episode；明确 GT pose + Oracle Semantic，非真实 VLM 零样本结果 |
| NaVid、Uni-NaVid、AwareVLN | `vln_reproduction/`；外部 `vln/reproduction/` | navid 新隔离环境；其他环境尚需完成 | 前期审计/测试/接口 smoke，不是三论文完整复现。9 月 16 日后台 NaVid 权重下载与依赖安装已成功结束；完整模型推理/正式评估未验收 |
| ROS2 窄通道记忆与安全融合 | `ros2_ws/src/narrow_memory_nav/` | 系统 ROS2 Humble，独立于模型 Conda | 迁移源码后重新 colcon build；默认观察模式，不自动控制机器人 |

正式 MP3D 的最新总入口是 `examples/narrow_passage_rl/results/formal_mp3d_summary_20260905/formal_experiment_work_report.md`；EAGOR 的最新入口是 `eagor_outputs/stop_diagnosis_20260911_161253/REPORT_CN.md`。早期 `reproducibility.md` 中“MP3D 0/11 缺失”等文字已经过时，不能据此重新下载或否定后来结果。

## 3. 必须搬走的六个目录

| 源目录 | 迁移包位置 | 当前约占用 | 作用 |
|---|---|---:|---|
| `/home/xiaotian/navigation/habitat-lab` | `files/habitat_lab/` | 23 GB | 全部修改、未跟踪文件、`.git`、结果、模型、ROS 源码 |
| `/home/xiaotian/vla` | `files/vla/` | 11 GB | NaVILA/Open-Nav/VLN-CE、Habitat-Lab/Sim 0.1.7 源码及构建资产 |
| `/home/xiaotian/navigation/narrow-passage-nav` | `files/narrow_passage_nav/` | 4.5 MB | 关联窄通道仓库，也保留避免遗漏 |
| `/media/xiaotian/ACD525D1B7A093D9/habitat_data` | `files/habitat_data/` | 35 GB | HM3D/版本化场景、窄通道数据、历代 RL checkpoint |
| `/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets` | `files/pointnav_assets/` | 5.2 GB | 官方 PointNav 数据/权重、派生 400/80 任务和 E2E 教师集 |
| `/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln` | `files/vln/` | 78 GB | MP3D、R2R/ObjectNav、NaVILA 权重、正式 VLN 结果、三方法源码和 NaVid 新权重 |

合计约 **152 GB，不含前三个 Conda 环境约 31 GB**。`vln/` 已含新 navid 环境约 5.4 GB；另打环境包会有重复。压缩率不保证，暂存、压缩包和解压文件会同时占空间；迁移盘与目标盘按实际方式留足余量，完整暂存+打包/解包宜准备 300 GB 以上可用空间。不要在当前只剩约 13 GB 的系统盘打包。

保留所有 checkpoint、CSV/JSON/NPZ、源代码快照、process ledger、视频和 `diagnostics/`。它们用于核验曾经排除的失败，不能只保留漂亮结果。`data_old/new_checkpoints/` 也在仓库内，勿漏。

仓库的 `data` 实际指向外盘 `habitat_data`，不是实体目录。导出时发现 1 个已有断链：`data_old/datasets/replica_cad/rearrange`，不要把它误认为迁移新产生的问题；需要 ReplicaCAD rearrange 时另外补齐。

## 4. 旧电脑：导出/打包

先停止**自己确认属于这些实验**的训练和结果写入，避免备份过程中模型或 CSV 变化。不要使用模糊 `pkill python`。当前核查时未发现上一轮 VLN 安装/下载仍在运行。

脚本默认只导出元数据：

```bash
cd /home/xiaotian/navigation/habitat-lab
python migration/prepare_migration.py \
  --output /media/xiaotian/ACD525D1B7A093D9/habitat_migration_snapshot_new
```

要实际制作完整本地迁移包，使用一个**不存在的新目录**：

```bash
python migration/prepare_migration.py \
  --output /media/xiaotian/ACD525D1B7A093D9/habitat_migration_full_20260918 \
  --copy-files --pack-envs
```

这条命令会花较长时间，生成 `files/`、`packed_envs/`、环境清单和指纹。前提是 `rsync` 与旧机已安装的 `conda-pack` 可用。脚本：

- 拒绝在源目录内部建立备份，避免自我递归；拒绝覆盖已有目标。
- 用 `rsync -a`，保留隐藏文件、未提交改动与软链接，**不加 `--delete`、不加 `-L`**。
- `conda-pack --ignore-editable-packages` 只是允许外部 editable 源码存在，**不意味着这些源码已装进环境包**；它们单独放在 `files/`，新机仍需路径兼容/重新绑定。
- 不使用 `--ignore-missing-files`；若环境打包失败，看 `pack_logs/`，不要强行忽略后宣称成功。
- Conda 环境导出不是验收通过证明；查看每条命令的 `.status.json`。

复制整个迁移包到目标机。可用移动硬盘或自己指定的 SSH/rsync；本次没有向远程机器上传文件。数据可能有许可限制，环境清单/工作区可能含私有路径或配置，勿直接公开发布。

若不制作中间大副本，也可分别用 `copy_commands.json` 的源路径直接传到新机的对应六个目录。传输完成后对每个目录做 `rsync -a --checksum --dry-run SOURCE/ DEST/` 核对；这会读取全体文件，耗时长，但比仅看文件大小更可靠。默认脚本的 24 项哈希**不覆盖全部 152 GB**。

## 5. 新电脑：恢复目录和绝对路径

示例将迁移包置于 `/data/habitat_migration`，并保持下面的六个目录名：

```text
/data/habitat_migration/
  files/{habitat_lab,vla,narrow_passage_nav,habitat_data,pointnav_assets,vln}/
  packed_envs/{habitat,navila,navila-eval,navid}.tar.gz
  migration_manifest.json
  critical_fingerprints.json
```

**最省事的初次复现方法是保留旧绝对路径的兼容链接**。不必与旧机同用户名，但管理员需要为这些路径创建父目录；目标路径必须不存在。下面的 `ln -sT` 故意没有 `-f`：遇到已有文件或目录立即停止人工核对，不覆盖。

```bash
sudo mkdir -p /home/xiaotian/navigation
sudo mkdir -p /media/xiaotian/ACD525D1B7A093D9/robot_nav_data
sudo ln -sT /data/habitat_migration/files/habitat_lab /home/xiaotian/navigation/habitat-lab
sudo ln -sT /data/habitat_migration/files/narrow_passage_nav /home/xiaotian/navigation/narrow-passage-nav
sudo ln -sT /data/habitat_migration/files/vla /home/xiaotian/vla
sudo ln -sT /data/habitat_migration/files/habitat_data /media/xiaotian/ACD525D1B7A093D9/habitat_data
sudo ln -sT /data/habitat_migration/files/pointnav_assets /media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets
sudo ln -sT /data/habitat_migration/files/vln /media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln
```

这些是**新机人工执行示例**，本轮未执行 sudo 或修改挂载点。替换示例 `/data/habitat_migration` 为你的实际迁移位置。若旧挂载点正在被别的数据盘占用，不执行这些链接；应选择单独容器/兼容挂载方案或逐项修改配置。

不想保留旧路径时，需要改活跃 YAML、命令参数、数据集 JSON/JSON.gz 中的 scene 路径和 editable `.pth/.egg-link`。**设置一个 `DATA_ROOT` 环境变量不会自动改掉所有硬编码路径**；特别是 `vln_reproduction/*.py` 和部分历史队列脚本。不要对整个项目做无差别替换，也不要改历史 provenance 和二进制 checkpoint。

## 6. 新电脑：恢复四套环境，不混装

先确保系统 GPU 驱动、EGL/OpenGL 可用，`nvidia-smi` 能识别设备。驱动属于新机系统，不在 Conda 包中；不要把旧机 `/usr/lib`、NVIDIA 驱动文件复制过去。当前 habitat 的 Torch 运行时是 CUDA 12.8，旧 VLN 环境是另一版本；以实际导出文件为准。

| 环境 | 用途 | 当前审计注意项 |
|---|---|---|
| habitat | Habitat 0.3.3、EAGOR、DEGNav、PointNav、SB3 | `pip check` 通过 |
| navila | NaVILA 模型与离线诊断 | `pip check` 报 decord 平台支持声明问题；需用实际导入/解码验收，勿盲目升级 |
| navila-eval | Habitat 0.1.7 + NaVILA/VLN-CE | 存在 tb-nightly/tensorflow 声明及 webdataset 版本冲突；是已有环境现状，不要自动覆盖已验证的兼容改动 |
| navid | 新建 Python 3.8 + NaVid 官方依赖 | 依赖安装已完成，decord 声明检查异常；完整 Habitat/模型运行尚未验收 |

在每个**全新空目录**解压环境，不能混到同一个 site-packages：

```bash
mkdir -p /data/habitat_migration/envs/habitat
tar -xzf /data/habitat_migration/packed_envs/habitat.tar.gz -C /data/habitat_migration/envs/habitat
/data/habitat_migration/envs/habitat/bin/python /data/habitat_migration/envs/habitat/bin/conda-unpack
```

对 `navila`、`navila-eval`、`navid` 各重复一次，换成对应包与目录。必须用**目标环境自身的 Python**运行 `conda-unpack`，然后 `source /data/habitat_migration/envs/habitat/bin/activate` 或直接调用该环境 `bin/python`。本手册后续使用绝对 Python 路径，避免新机 `conda run -n habitat` 找到同名但错误的环境。

也可使用新增的新机辅助脚本，替代上面四次手动解压。先完成第 5 节路径恢复；不要在旧机运行 `restore`，不要与手动解压重复执行：

```bash
cd /data/habitat_migration/files/habitat_lab
bash migration/new_machine.sh plan /data/habitat_migration
bash migration/new_machine.sh restore /data/habitat_migration
bash migration/new_machine.sh check /data/habitat_migration
```

`plan` 只显示步骤；`restore` 拒绝已有环境目录，不覆盖或自动删除；中途失败会保留部分目录供检查，不能直接重复运行来覆盖。`check` 执行四套环境的实际 CUDA tensor 运算，并打印两套 Habitat 的导入位置，遇到错误立即退出。脚本不会安装驱动、升级依赖或自动建立系统路径链接。**这些检查不是 EGL 渲染或模型完整推理验收**，仍需第 7 节 smoke。本轮仅在旧机验证脚本语法与无写入的 plan 模式，没有执行新机恢复或新机 GPU 验收。

如果某脚本仍硬编码 `/home/xiaotian/miniconda3/envs/.../bin/python`，需要在不存在该路径时创建相应兼容链接，或者修改该脚本的 Python 参数。不能只改变 shell 的当前激活环境。

editable 源码绑定核查：

```bash
/data/habitat_migration/envs/habitat/bin/python -c 'import habitat,habitat_baselines; print(habitat.__version__,habitat.__file__,habitat_baselines.__file__)'
/data/habitat_migration/envs/navila-eval/bin/python -c 'import habitat,habitat_sim; print(habitat.__version__,habitat.__file__,habitat_sim.__file__)'
```

前者应来自迁移后的本仓库 0.3.3，后者应来自 `vla/habitat-lab-v0.1.7`，不能交叉。若 editable 路径未恢复，使用对应环境的 `pip install --no-deps -e 正确源码目录` 重新绑定；**不要同时重装依赖**，避免冲掉 Transformers/DeepSpeed 等已有兼容补丁。旧 Habitat-Sim 的 `.egg` 内有原生库，若新系统 ABI/GPU 不兼容需用保留源码重建，不能仅复制 `.egg-link`。

备选在线重建：`conda create -p 新环境 --file environments/habitat/conda-explicit.txt` 只能恢复 Conda 管理部分；pip 包、editable 源码、手工补丁还需恢复。`environment-full.txt` 可作 YAML 参考，先处理旧 `prefix:` 和本地路径。不要假定单独 `pip install -r pip-freeze.txt` 就能重建所有修改。

## 7. 分层验收与精确复现入口

### 7.1 先查文件，不跑长实验

```bash
python3 /data/habitat_migration/files/habitat_lab/migration/verify_migration.py \
  --snapshot /data/habitat_migration \
  --files-root /data/habitat_migration/files
```

期望 24 项关键指纹全部通过。随后检查 `source_symlinks.json` 和 `readlink -f .../habitat_lab/data`。不要重新生成 episodes 替代原数据；重新挖掘会改变任务集合。

后续统一从仓库根目录执行，支持指定输出的脚本写入全新 `migration_runs/`；第 7.7 节接口 smoke 使用脚本自身的时间戳新目录，不覆盖论文产物：

```bash
cd /home/xiaotian/navigation/habitat-lab
mkdir -p migration_runs
HAB_PY=/data/habitat_migration/envs/habitat/bin/python
VLN_PY=/data/habitat_migration/envs/navila-eval/bin/python
VLN_ROOT=/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln
PN_ROOT=/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets
```

### 7.2 EAGOR

```bash
"$HAB_PY" -m eagor_repro.scripts.run_tests
"$HAB_PY" -m eagor_repro.scripts.evaluate_eagor --overlay configs/eagor/mp3d_oracle.yaml --preflight
"$HAB_PY" -m eagor_repro.scripts.run_stop_matrix --root migration_runs/eagor_stop_01 --phase smoke --workers 1
# smoke 正确后才跑 100 episode
"$HAB_PY" -m eagor_repro.scripts.run_stop_matrix --root migration_runs/eagor_stop_01 --phase matrix --workers 1
"$HAB_PY" -m eagor_repro.scripts.analyze_stop_matrix --root migration_runs/eagor_stop_01
```

本轮旧机重新运行 76/76 测试通过。必须保留 `eagor_outputs/online_navigation_20260911/episode_manifest.json`，队列会读取这一原始 10 任务清单；只复制 `eagor_repro/` 不够。保留 GT pose、Oracle Semantic、Area/Oracle Stop 标签；不能将其作为普通 VLN 成绩。

其余 EAGOR 球谐压力测试、长时序、failure attribution、视频命令见 `eagor_repro/README.md`、`README_GROUP_MEETING_CN.md` 和最新停止报告。不要用 `run_smoke_test` 的短场景替代正式 10 任务。

### 7.3 程序化窄通道/记忆与消融

```bash
"$HAB_PY" examples/narrow_passage_rl/eval_harder_benchmark.py \
  --episodes 10 --seed 42 --methods rule_baseline geometry_fsm \
  --output-csv migration_runs/procedural_smoke.csv \
  --output-summary migration_runs/procedural_smoke_summary.csv
```

这是迁移 smoke，不是全部论文设置。各正式数据/seed/variant 从 `results/narrow_passage_rl/results_manifest.yaml`、`paper_ready_results.md`、`configs/` 和原 `resolved_config.json` 读取。相关 `train_sb3_v2.py`、`train_recurrent_ppo_v2.py`、`train_belief_mode_ppo.py`、`train_mode_selector_*`、记忆/消融脚本均随完整仓库迁移；不要把不同轮次的 checkpoint 混在同一表中。

### 7.4 官方 PointNav

```bash
"$HAB_PY" examples/narrow_passage_rl/eval_official_pointnav_baselines.py \
  --method pointnav_ppo --num-episodes 3 --seed 1701 \
  --data-path "$PN_ROOT/mp3d_v1/val/val.json.gz" \
  --scenes-dir "$VLN_ROOT/data/scene_datasets" \
  --ppo-checkpoint "$PN_ROOT/checkpoints/mp3d-rgbd-best.pth" \
  --output-dir migration_runs/pointnav_smoke
```

通过后去掉 `--num-episodes 3`，使用 `--seed 1701 --seed 1702 --seed 1703` 复现学习策略的 495×3；DD-PPO 改 `--method ddppo` 并指定 `--ddppo-checkpoint "$VLN_ROOT/data/ddppo-models/gibson-2plus-resnet50.pth"`。七种基线和种子安排见 `run_official_pointnav_sequence.sh`。该旧队列含旧输出路径/跳过逻辑，**不要直接在迁移的历史结果目录重跑并将 SKIP 视为新实验通过**。

### 7.5 DEGNAV-E2E 与派生窄通道

先复用已训练权重验收：

```bash
"$HAB_PY" examples/narrow_passage_rl/eval_degnav_e2e.py \
  --checkpoint examples/narrow_passage_rl/results/degnav_e2e_mp3d_narrow_v1_20260904/seed1701/train/best.pt \
  --dataset "$PN_ROOT/mp3d_narrow_v1/val/val.json.gz" \
  --num-episodes 3 --seed 1701 --output-dir migration_runs/e2e_smoke
```

重新训练 seed1701：

```bash
"$HAB_PY" examples/narrow_passage_rl/train_degnav_e2e_bc.py \
  --data-root "$PN_ROOT/mp3d_narrow_v1/e2e_teacher" --input-type depth \
  --output-dir migration_runs/e2e_seed1701/train --epochs 20 --batch-size 2 \
  --sequence-length 16 --stride 8 --hidden-size 256 \
  --class-balance-power 1.0 --seed 1701
```

seed1702/1703 使用各自输出目录；全量评估去掉 3-episode 限制，保持相同 80 val。保留教师 NPZ 的 `history_storage_version=2` 和配置；不能用旧的未来信息泄漏版本替代。若需重采集，严格按 `run_degnav_e2e_sequence.sh` 的 400/80、seed1701、Depth128、半径0.18 配置重跑。官方 PointNav 与派生窄通道协议不同，分表比较。

### 7.6 NaVILA / VLN-CE

```bash
"$VLN_PY" examples/narrow_passage_rl/vln_r2r_preflight.py --help
"$VLN_PY" examples/narrow_passage_rl/vln_r2r_preflight.py --output-dir migration_runs/vln_preflight
cd /home/xiaotian/vla/NaVILA/evaluation
PYTHONPATH=/home/xiaotian/navigation/habitat-lab "$VLN_PY" run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
  --run-type eval --num-chunks 1 --chunk-idx 0 \
  EVAL_CKPT_PATH_DIR "$VLN_ROOT/models/navila-llama3-8b-8f" \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT 1 \
  RESULTS_DIR /home/xiaotian/navigation/habitat-lab/migration_runs/navila_smoke \
  VIDEO_OPTION "[]"
```

保留已修改的 NaVILA 源码；当前 trainer 的 4-bit 与安全适配器实现不一定存在于官方 clone。正式历史队列是 10 shards，逐 chunk 0..9，`EVAL.EPISODE_COUNT -1`；另做适配器条件时显式 `EVAL.DEGNAV_ADAPTER.ENABLED True`，换新输出目录。使用官方 `scripts/eval_jsons.py` 汇总前检查任务去重与 shard 完整性；参考旧 `formal_runs/`，不是混入迁移 smoke。

### 7.7 新三方法与 ROS

```bash
cd /home/xiaotian/navigation/habitat-lab
"$VLN_PY" -m unittest discover -s vln_reproduction -p 'test_*.py' -v
"$VLN_PY" vln_reproduction/simulator_smoke.py --method navid
"$VLN_PY" vln_reproduction/simulator_smoke.py --method uninavid
```

这两段视频只是固定动作接口验证，**不是 NaVid/Uni-NaVid 模型导航视频**。输出在 `$VLN_ROOT/reproduction/smoke/<新时间戳>_<method>/`，不是 `migration_runs/`；已有 audit 任务文件必须随数据迁移。9 月 18 日确认 NaVid 两个完整权重 shard 与下载校验存在、安装进程退出码0；先前 `REPRO_STATE.md`/CSV 关于“下载未完成”的描述是中途快照，应以 `vln/reproduction/attempts/*/result.json` 和 `assets/navid/download_verified.json` 为准。此前显存字节分配探测 OOM 不是完整模型加载 OOM；新机应重新做真实模型加载和推理，不能直接宣称三模型已复现。

ROS2 仅恢复 `ros2_ws/src/` 的构建用途，在 Humble 系统重新 `colcon build --symlink-install`，不要 source 旧 `build/install/log` 产物来冒充新机编译。先保持 `safety_fusion_node.enabled=false`，按其 README 观察话题；未现场确认/授权时不启用 `/cmd_vel` 输出。Jetson Foxy 是另一套部署目标，不使用这些桌面 Conda 包。

## 8. 迁移验收清单

- [ ] 六个目录和四个环境归档齐全；pack/copy 命令无未处理失败。
- [ ] 24 个关键哈希通过；完整文件传输另做 checksum/dry-run 核对。
- [ ] `data`、模型、旧 Habitat 源码软链接正确；已有 ReplicaCAD 断链单独记录。
- [ ] `habitat` 导入0.3.3、`navila-eval` 导入0.1.7，模块位置正确。
- [ ] 新机 EGL 能真实渲染；前进/转向单位正确，画面不黑。
- [ ] EAGOR 76 测试通过；新三方法契约测试通过但不计为模型完成。
- [ ] PointNav/E2E/NaVILA 各先 smoke，再重跑需要的完整协议。
- [ ] 新结果不覆盖旧结果；使用同任务、种子、相机、动作预算、精度、停止定义。
- [ ] 迁移训练是“从头重新训练”还是“继续训练”明确区分；`best.pt` 可能不足以恢复优化器/RNG/采样器。恢复中断训练必须使用完整 checkpoint，并核验对应脚本 resume 支持，不能只加载 actor 就声称无缝续训。

同种子跨 GPU、驱动和算子版本也不保证逐位一致。验收应先看模块/任务/协议一致，再看逐任务输出和差异原因，不以单个成功率接近代替复现。
