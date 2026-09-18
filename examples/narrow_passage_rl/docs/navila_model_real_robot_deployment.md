# NaVILA 模型说明、真实机器人部署与位姿变化训练方案

> 文档日期：2026-08-20  
> 适用项目：DEGNav / Habitat-Lab narrow-passage / DeepRobotics Lite3  
> 模型：`navila-llama3-8b-8f`  
> 文档原则：已运行结果与工程建议严格分开；不将离线接口测试描述为实机闭环或标准 VLN 成绩。

## 1. 结论摘要

当前 NaVILA checkpoint 是一个约 **8.494B 参数**的视觉语言导航模型，由
Llama-3 8B、SigLIP 视觉塔和多模态 projector 组成。模型以自然语言指令和
最近 8 帧 RGB 为输入，当前接口通过自回归文本生成下一步导航提议，例如
`move forward 75 cm`、`turn left 30 degree` 或 `stop`。

在 NVIDIA RTX A4000 上使用 bitsandbytes NF4 4-bit 推理时，保存的 640-case
诊断得到平均 `1.466 s/次`、P95 `1.485 s/次`，约为 `0.68` 次高层决策每秒；
模型加载后显存约 `6.31 GiB`，推理峰值约 `7.88 GiB`。

当前模型不应直接控制真实四足机器人。已测结果显示它存在明显的
`move_forward` 先验、停止失败、左右方向 grounding 较弱以及视觉扰动敏感性。
推荐的实机系统是分层结构：

```text
NaVILA semantic proposal / local waypoint
                    |
                    v
DEGNav geometry belief + failure memory + safety adapter
                    |
                    v
Mode-conditioned local controller
                    |
                    v
Lite3 locomotion controller + independent watchdog / E-stop
```

相机安装角度或位置变化首先是外参与时钟标定问题，不能仅依靠训练补偿。
训练应进一步采用真实三维相机外参随机化、机器人中心坐标航点、初始 yaw/offset
随机化、显式外参输入和按场景/安装方案隔离的数据划分。

## 2. 当前模型及文件

当前 checkpoint：

```text
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f
```

当前诊断使用的 NaVILA 源码版本：

```text
76b98f233dd0fff05dfcd69435eec6740febff9d
```

主要模型文件：

```text
navila-llama3-8b-8f/
├── config.json
├── llm/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── model-00001-of-00004.safetensors ... model-00004-of-00004.safetensors
├── vision_tower/
│   ├── config.json
│   └── model.safetensors
└── mm_projector/
    ├── config.json
    └── model.safetensors
```

## 3. 模型结构详解

### 3.1 总体数据流

```text
Language instruction
        +
8-frame RGB history
        |
        v
SigLIP vision tower
        |
        v
Multimodal projector
        |
        v
Visual tokens + language tokens
        |
        v
Llama-3 autoregressive decoder
        |
        v
Text action proposal
        |
        v
Regex parser -> move_forward / turn_left / turn_right / stop
```

当前 NaVILA 模型本身不直接读取 DEGNav 的 depth、clearance、OBB morphology、
failure memory 或接触状态。这些信息进入模型下方的几何与安全决策层。

### 3.2 视觉编码器

视觉塔配置为 `SiglipVisionModel`：

| 配置 | 数值 |
|:---|---:|
| 输入分辨率 | 384 × 384 |
| patch size | 14 |
| hidden size | 1152 |
| intermediate size | 4304 |
| Transformer layers | 27 |
| attention heads | 16 |
| 参数量 | 428,225,600 |

每帧 RGB 经 SigLIP 转换为视觉 token。当前推理 prompt 包含 8 个图像位置，
对应 7 帧历史观测和 1 帧当前观测。时序关系主要由图像 token 在 Llama 上下文
中的顺序表达；当前没有单独向真实机器人提供精确帧间里程计或相机外参 token。

### 3.3 多模态 projector

当前 projector 类型为 `mlp_downsample`：

| 配置 | 数值 |
|:---|---:|
| 视觉维度 | 1152 |
| LLM hidden size | 4096 |
| 参数量 | 35,668,992 |

projector 负责将 SigLIP 特征转换到 Llama-3 token 空间。对于相机域适配和新增
waypoint head，它是优先考虑微调的模块，因为成本远低于全参数微调 8B LLM。

### 3.4 语言模型

语言主干为 `LlamaForCausalLM`：

| 配置 | 数值 |
|:---|---:|
| hidden size | 4096 |
| Transformer layers | 32 |
| attention heads | 32 |
| key/value heads | 8 |
| intermediate size | 14336 |
| vocabulary size | 128259 |
| checkpoint max position | 8192 |
| 当前 loader context | 2048 |
| 参数量 | 8,030,285,824 |

推理使用 greedy decoding：`do_sample=False`、`temperature=0`、
`max_new_tokens=32`、`use_cache=True`。

### 3.5 总参数量

参数量通过本地 safetensors header 中的 tensor shape 直接累计，而不是仅根据
“8B”名称估计：

| 组件 | Tensor 数 | 参数量 | 比例 |
|:---|---:|---:|---:|
| Llama-3 LLM | 291 | 8,030,285,824 | 94.54% |
| SigLIP vision tower | 448 | 428,225,600 | 5.04% |
| multimodal projector | 6 | 35,668,992 | 0.42% |
| **总计** | **745** | **8,494,180,416** | **100%** |

BF16 checkpoint 总大小约 `16.998 GB`，即 `15.83 GiB`。当前运行时使用 NF4
4-bit 权重量化，因此 GPU 显存明显小于 BF16 checkpoint 大小。

## 4. 当前推理接口

当前 prompt 的核心语义是：给定历史视频和当前图像，根据语言任务输出下一步
动作；允许左转、右转、前进一定距离或任务完成时停止。

输出经规则解析为：

```text
move_forward(value in cm)
turn_left(value in degree)
turn_right(value in degree)
stop
```

当前 parser 只是文本到动作提议的接口，不是物理安全控制器。DEGNav safety
adapter 会进一步限制：

- 最大前进距离：75 cm；
- 最大旋转角：90°；
- 不确定时将前进距离限制到 20 cm；
- collision 或高 stuck score 优先进入 Recover；
- 低 `p_feas`、低 clearance、低 body margin 或高 memory risk 优先 Reject；
- 只有显式标记为 trusted control directive 的立即停止指令才可直接覆盖模型。

## 5. 推理速度和资源占用

### 5.1 运行环境

| 项目 | 配置 |
|:---|:---|
| GPU | NVIDIA RTX A4000，16 GiB |
| PyTorch | 2.3.0 + CUDA 12.1 |
| Transformers | 4.37.2 |
| bitsandbytes | 0.41.0 |
| flash-attn | 2.5.8 |
| 量化 | NF4 4-bit |
| 输入 | 8 帧 RGB |
| 平均输出 | 约 12 token |

### 5.2 已保存测量

| 测试 | Cases | Mean | Median | P95 | Peak GPU |
|:---|---:|---:|---:|---:|---:|
| dataset-free smoke | 7 | 1.409 s | 1.337 s | 1.696 s | 8003 MiB |
| controlled diagnostic | 640 | 1.466 s | 1.467 s | 1.485 s | 8067 MiB |

由 640-case 结果得到：

```text
high-level decision rate ~= 1 / 1.466 = 0.68 Hz
rough output-token rate ~= 12.02 / 1.466 = 8.2 token/s
model load time ~= 4.74 s
GPU memory after load ~= 6.31 GiB
peak allocated GPU memory ~= 7.88 GiB
```

延迟使用 `torch.cuda.synchronize()` 包围 `model.generate()` 测量，包括模型内部
视觉编码、语言 prefill 和生成，但不包括从磁盘读取图像、PIL/processor 预处理、
ROS2 transport、定位、深度处理、局部控制和机器人执行。因此它不是实机端到端
控制周期。

## 6. 已验证能力和已知问题

### 6.1 已验证内容

- checkpoint 文件完整并可在本地加载；
- 4-bit 推理成功；
- 7/7 smoke case 完成；
- 640-case 输出均可解析为动作；
- 重复输入的 deterministic output 一致；
- NaVILA-to-DEGNav safety adapter 的 Commit/Explore/Recover/Reject 接口已通过
  离线单元测试和 saved-output replay。

### 6.2 当前模型问题

640-case 诊断显示：

| 问题 | 证据 |
|:---|:---|
| 前进动作先验 | 256 个 R2R 文本探针中 71.5% 输出 move-forward |
| 停止失败 | explicit stop override compliance 仅 1.6% |
| 左右方向 grounding 弱 | 受控 turn-left/right direct match 均为 0% |
| 障碍物 veto 缺失 | blocked synthetic fixture 仍可能输出前进 |
| 视觉扰动敏感 | 非 clean 扰动 action consistency 约 75% |
| 时间顺序敏感 | temporal reverse consistency 62.5% |

R2R 文本探针使用真实 val-unseen instruction，但视觉帧与 R2R 场景不匹配。
它只能说明动作先验和接口缺陷，不能报告 SR、SPL、NE、nDTW 或未见场景泛化。
标准 R2R/VLN-CE 评测当前仍缺少受许可的 MP3D scene assets。

## 7. 真实机器人推荐部署架构

### 7.1 时间尺度分层

NaVILA 的速度约为 0.68 Hz，远低于四足稳定控制所需频率。因此必须异步分层：

| 层 | 建议更新频率 | 输出 |
|:---|:---:|:---|
| RGB/Depth/LiDAR sensing | 15–30 Hz，按硬件实测 | 时间同步观测 |
| NaVILA semantic proposal | 0.5–0.7 Hz | 语义目标或短时航点 |
| DEGNav geometry/safety | 10–20 Hz | feasibility 和 mode |
| local velocity controller | 10–30 Hz | bounded `vx, vy, wz` |
| Lite3 locomotion controller | 50–200 Hz，按实际 controller | joint/torque target |
| independent watchdog/E-stop | 独立硬实时或高优先级 | zero command / disable |

上述频率是部署设计范围，不是当前仓库已经测得的 Lite3 实机配置。实际值必须从
机器人 SDK、locomotion checkpoint、传感器频率和 rosbag latency 中确定。

### 7.2 ROS2 节点建议

| 节点 | 输入 | 输出 |
|:---|:---|:---|
| `realsense2_camera` | D435i | RGB、depth、camera_info |
| Livox driver | MID360 packets | point cloud、IMU |
| localization | LiDAR/IMU/odometry | `map -> odom -> base_link` |
| `navila_node` | RGB history、language | proposal、timestamp、confidence |
| `geometry_node` | depth/cloud、pose、morphology | clearance、`p_feas`、LCB/UCB |
| `safety_adapter_node` | proposal + geometry + memory | mode、safe local target |
| `local_controller` | mode + local waypoint | bounded velocity |
| `lite3_bridge` | velocity/skill | vendor SDK or locomotion command |
| `command_watchdog` | command timestamps、heartbeat | stop/hold |
| `experiment_logger` | relevant topics | rosbag、CSV、video |

当前仓库已有离线 NaVILA 推理和 deterministic safety adapter，但没有完整、
经过实机复现的 `navila_node -> geometry_node -> Lite3 bridge` ROS2 闭环实现。

### 7.3 推荐运行接口

生产接口应从“文本速度动作”升级为：

```text
semantic_goal
waypoint_xy              # base_link frame
waypoint_yaw
waypoint_covariance
p_complete
proposal_timestamp
source_frame_ids
```

下游必须检查：

- proposal 是否超时；
- RGB/depth/pose 时间戳是否一致；
- waypoint 是否在机器人坐标系；
- 当前几何是否可通行；
- robot morphology 与安全 margin 是否正确；
- collision、stuck、memory risk 是否要求 Recover/Reject；
- operator E-stop 是否优先。

由于单次 VLN 推理约 1.5 秒，机器人不能在等待期间盲目前进。局部控制器只能在
短时安全 horizon 内跟踪最近的已验证航点；超时 proposal 必须丢弃或重新验证。

### 7.4 计算硬件

当前峰值显存约 7.88 GiB。考虑 CUDA context、ROS2 图像缓存、深度网络和显存
碎片，建议：

- 最低可用 GPU 显存：约 10 GiB；
- 更稳妥：12–16 GiB；
- 当前已验证平台：RTX A4000 16 GiB；
- 如果机载 GPU 不足，可使用有线随行计算机，但必须有断线 watchdog，不能让网络
  中断后继续执行历史命令。

## 8. 坐标系、外参与时间同步

### 8.1 必须标定的量

```text
camera intrinsics
RGB-depth alignment
T_base_camera
T_base_lidar
T_imu_base
map -> odom -> base_link TF
RGB/depth/LiDAR/IMU timestamp offsets
robot body and leg collision envelope
```

相机坐标系中的预测点 `p_camera` 必须转换为机器人坐标系：

```math
p_{base} = T_{base \leftarrow camera} p_{camera}
```

如果需要全局坐标：

```math
p_{map} = T_{map \leftarrow base}(t) p_{base}
```

局部控制器应优先接收 `base_link` 下的相对航点。这样机器人初始世界位置改变时，
控制接口本身仍保持一致。

### 8.2 标定与训练的边界

- 固定安装但外参未知：先标定，通常不应通过训练猜测坐标变换。
- 固定安装与训练相机不同：标定后仍可能有视觉 domain gap，可做小规模适配。
- 安装会重复拆装或动态变化：除标定外，模型应显式接收外参并进行外参随机化训练。
- quadruped 行走导致动态 roll/pitch：使用 IMU 做 gravity alignment，并在训练中覆盖
  实际姿态分布。
- 相机平移会产生视差，简单 rotate/crop/affine augmentation 不能正确模拟，应使用
  三维渲染、深度重投影或真实多安装位数据。

## 9. 不同相机角度和安装位置怎么训练

### 9.1 外参随机化

设标称相机外参为 `T_base_camera`，训练时使用：

```math
T'_{base,camera} = \exp(\hat{\xi}) T_{base,camera}
```

其中 `xi` 表示三维平移与 roll/pitch/yaw 扰动。扰动范围必须来自真实装配误差和
行走姿态日志，不应为了“增强更多”而任意扩大。预实验可采用以下分层网格，再按
实测分布修订：

- camera yaw/pitch：标称值附近 `±5° / ±10° / ±15°`；
- camera roll：由机身姿态和安装误差确定；
- camera height/forward/lateral translation：标称值附近 `±1–5 cm`；
- 动态机身 roll/pitch：从 IMU rosbag 分位数采样；
- camera latency：从实际 RGB 到 pose/command 的时间差采样。

### 9.2 显式输入外参

如果部署中可能存在多个相机安装方案，仅做随机化会迫使模型隐式猜测外参，存在
不可辨识性。更可靠的方案是向 waypoint/projector head 输入：

```text
camera height
camera translation xyz
camera roll/pitch/yaw or 6D rotation
intrinsics fx, fy, cx, cy
IMU gravity vector
```

可以将外参编码为一个 calibration token，或与 geometry/waypoint head 的输入拼接。
所有标签仍统一定义在 `base_link`。

### 9.3 真实数据适配

建议在同一批真实场景中采集：

- 标称安装位；
- 允许误差范围内的多个安装位；
- 不同机身 roll/pitch；
- 同一语言目标的多视角接近轨迹；
- 相同状态下的 open/blocked、goal-not-complete/goal-complete 配对。

训练集和验证集不能按帧随机切分。应至少做 `leave-one-mount-out`：留出一个完整
相机安装方案，只用于验证泛化。

## 10. 不同机器人初始位置和朝向怎么训练

### 10.1 使用机器人中心坐标

避免预测绝对世界坐标。参考航点应转换为当前机器人坐标：

```math
w_t^{base} = T_{base \leftarrow map}(t) w_t^{map}
```

朝向输入建议使用：

```text
sin(yaw_error), cos(yaw_error)
```

而不是直接回归存在 `-pi/pi` 跳变的角度。

### 10.2 初始位姿随机化

训练场景应覆盖：

- 多个起点距离；
- 左右 lateral offset；
- 正负 yaw；
- 前后偏移；
- 入口部分遮挡；
- 多条等价路线；
- 目标附近的完成与未完成状态。

可复用 strict feasibility stress 轴：

```text
yaw = {0, 15, 30, 45, 60} degrees，并加入左右符号
lateral offset = {0, 0.10, 0.20} m，并加入左右符号
noise = {clean, mild, severe}
```

训练时可从连续分布采样，评测时保留固定 bins，确保不同方法严格配对。

### 10.3 对称与反事实样本

为解决当前左右 grounding 弱的问题，应构造：

- 同一场景左右镜像，标签同步交换 left/right；
- 同一视觉历史配对 left/right 指令；
- 同一轨迹在转向前、转向后分别标注；
- “经过沙发后左转”与“现在立即左转”的 future/immediate 对比；
- open/blocked 视觉对，保持前进语言不变；
- goal 前与 goal 后的 stop counterfactual。

注意：镜像图像时必须同步更新 waypoint、yaw、相机外参和所有几何标签。

## 11. 推荐训练数据格式

每个样本至少保存：

```text
episode_id
scene_id
mount_id
timestamp and frame timestamps
language instruction
8-frame RGB history
depth or point cloud reference
camera intrinsics
T_base_camera
T_map_base / odometry
IMU roll/pitch/yaw and gravity
reference waypoint in base_link
reference waypoint covariance or valid region
reference local action
goal-complete label
passage/clearance/feasibility label
collision/contact/stuck outcome
robot morphology
```

训练数据需要平衡：

- `move_forward / turn_left / turn_right / stop`；
- 正负 yaw 与左右 offset；
- clean 和每类视觉扰动；
- open 和 blocked；
- goal complete 和 incomplete；
- nominal 和不同 camera mount；
- seen scene 和 held-out scene。

## 12. 推荐训练路线

### 阶段 0：不训练，先完成标定与闭环接口

1. 标定 intrinsics/extrinsics 和时间偏差。
2. rosbag 离线运行 NaVILA、geometry 与 adapter。
3. 检查 TF、proposal timestamp 和安全停止。
4. 建立当前 checkpoint 的真实数据 baseline。

这一步用于区分“坐标错误”和“模型泛化错误”。如果 TF 方向或时间戳错误，继续
微调模型只会学习补偿某一套错误标定。

### 阶段 1：训练 waypoint/completion head

生产版本不建议继续以自然语言距离/角度作为主要控制接口。推荐输出：

```text
z_semantic
waypoint_mean_xy
waypoint_yaw
waypoint_covariance
p_complete
```

先冻结 SigLIP 和 Llama-3，训练 projector 后部与 waypoint/completion heads。

### 阶段 2：LoRA 适配

如果 waypoint head 无法解决语言 grounding：

- 对 Llama 后若干层使用 LoRA；
- 保持大部分 LLM 和 vision tower 冻结；
- 使用动作平衡和 counterfactual 数据；
- 单独报告每个动作、stop recall 和 waypoint error；
- 不用训练集 Success 选择 checkpoint。

### 阶段 3：有限视觉微调

只有 held-out mount/scene 仍显示稳定视觉 domain gap 时，才考虑解冻 projector
或少量 vision layers。必须监控原有 clean/val-unseen 性能，防止真实小数据过拟合。

### 阶段 4：低层 locomotion 独立训练

NaVILA 不学习关节级稳定控制。Lite3 locomotion policy 应独立覆盖：

- velocity tracking；
- IMU 和关节噪声；
- 接触与摩擦变化；
- actuator delay；
- 质量和惯量变化；
- 外力扰动；
- camera-body extrinsic error；
- 低速转向、窄通道和 Recover primitive。

高层只输出 waypoint/mode/velocity reference，低层策略处理姿态、足端接触和关节动作。

## 13. 推荐损失函数

```math
L =
  \lambda_{wp} L_{waypoint}
  + \lambda_{yaw} L_{yaw}
  + \lambda_{act} L_{action}
  + \lambda_{stop} L_{completion}
  + \lambda_{unc} L_{waypoint\_NLL}
  + \lambda_{cons} L_{view\_consistency}
  + \lambda_{risk} L_{risk}
```

建议定义：

| Loss | 作用 |
|:---|:---|
| `L_waypoint` | base-frame waypoint 的 Huber/L1 误差 |
| `L_yaw` | `sin/cos` 朝向误差 |
| `L_action` | 平衡后的 forward/left/right/stop CE |
| `L_completion` | 独立 stop/completion BCE 或 focal loss |
| `L_waypoint_NLL` | waypoint 均值和 covariance 的 Gaussian NLL |
| `L_view_consistency` | 合理外参/视觉扰动下的输出一致性 |
| `L_risk` | collision/infeasible 辅助监督；不替代 DEGNav 几何安全门 |

权重必须通过独立 validation split 确定，不能在最终 test/mount/scene 上调参。

## 14. 数据增强与域随机化

### 14.1 视觉

- 曝光和 white balance；
- motion blur、Gaussian/shot noise；
- 中心/边缘遮挡；
- RGB compression；
- frame dropout 和 duplicated frame；
- variable history length；
- temporal jitter；
- RGB-depth timestamp mismatch。

### 14.2 几何与位姿

- camera SE(3) extrinsic perturbation；
- robot start translation/yaw；
- odometry drift；
- IMU roll/pitch bias；
- depth scale/no-return/invalid sectors；
- morphology 和 safety margin 作为显式条件，不可静默改变标签。

### 14.3 不应采用的替代方法

- 不能用单纯 2D rotate/crop 模拟相机三维平移；
- 不能把校准错误当成随机噪声长期保留；
- 不能只增加训练时长解决动作塌缩；
- 不能只扩充 forward 样本；
- 不能让 NaVILA 直接学习绕过 OBB collision gate；
- 不能使用 test scene、test mount 或最终实机 trial 调超参数。

## 15. 数据划分与评测

推荐至少四级隔离：

```text
train scenes + train mounts
validation scenes + train mounts
held-out scenes + train mounts
held-out scenes + held-out mount
```

真实机器评测应按固定起点/目标/障碍布置进行配对，方法顺序随机化，并保存
rosbag、外部视频和 episode CSV。

### 15.1 模型指标

- waypoint position/yaw error；
- waypoint NLL 和 covariance coverage；
- per-action confusion matrix；
- stop precision/recall、ECE/Brier；
- left/right counterfactual consistency；
- mount-wise、yaw-wise、offset-wise metrics；
- inference latency、stale proposal rate、GPU memory。

### 15.2 导航指标

- Success、SPL、Navigation Error；
- Collision、Near-collision、Timeout/Stuck；
- Oscillation、Recovery success；
- Wrong decision 和 Human intervention；
- End-to-end latency；
- 不同 camera mount、start yaw、lateral offset、noise 的分组结果。

### 15.3 真实机器人验收建议

1. 标定误差在允许范围内；
2. 所有 waypoint 能正确转换到 `base_link`；
3. stale proposal 不会继续驱动机器人；
4. collision/stuck/E-stop 始终覆盖 NaVILA；
5. open-space waypoint tracking 先通过；
6. 不同 yaw/offset 通过后再进入 doorway；
7. doorway 通过后再测试 narrow passage；
8. 每种方法、场景和 mount 使用配对 trial；
9. 至少三个训练 seed，并报告置信区间；
10. 任何实机安全结论必须由接触、clearance 和人工接管记录支持。

## 16. 分阶段部署顺序

```text
1. Offline rosbag replay
2. Static robot perception/TF validation
3. Command bridge with motors disabled or robot suspended
4. Open-space low-speed waypoint tracking
5. Different start yaw/lateral offsets
6. Doorway with automatic Commit disabled
7. Enable geometry Commit/Explore/Recover/Reject
8. Narrow-passage trials
9. Full language-goal navigation
10. Repeated trials with complete logs and external video
```

每一阶段失败都应停止进入下一阶段。安全员、物理急停、速度上限和人工接管规则
必须在实验前固定。

## 17. 当前项目状态与下一步

| 项目 | 当前状态 |
|:---|:---|
| NaVILA checkpoint/runtime | 已验证 |
| 4-bit A4000 推理速度/显存 | 已测量 |
| 640-case 离线诊断 | 已完成 |
| deterministic safety adapter | 离线 smoke 已完成 |
| 标准 R2R/VLN-CE | 未运行，缺 MP3D scene assets |
| learned waypoint/completion head | 尚未实现 |
| RGB-depth semantic/geometry fusion | 尚未训练 |
| failure memory 与 NaVILA 闭环 | 尚未实现 |
| ROS2 NaVILA-to-Lite3 闭环 | 尚未形成可复现实现 |
| Lite3 VLN 实机统计 | 尚未归档 rosbag、多次 trial 和完整指标 |

优先级建议：

1. 完成 `T_base_camera`、时间同步和 rosbag 基线；
2. 实现 base-frame waypoint/completion 输出；
3. 接通 ROS2 geometry/safety adapter，但保持 motors disabled 测试；
4. 采集多 yaw、offset、camera mount 的配对数据；
5. 冻结大模型训练 waypoint/completion head；
6. 仅在 held-out mount/scene 需要时加入 LoRA；
7. 完成 open-space、doorway、narrow-passage 分阶段实机验证。

## 18. 当前复现命令和证据路径

### 模型运行 smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/narrow_passage_rl/vln_dataset_free_smoke.py \
  --output-dir examples/narrow_passage_rl/results/vln_dataset_free_NEW_TIMESTAMP
```

### 640-case 离线诊断

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/narrow_passage_rl/vln_batch_diagnostic.py \
  --r2r-samples 256 \
  --output-dir examples/narrow_passage_rl/results/vln_batch_diagnostic_NEW_TIMESTAMP
```

### Safety adapter

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  examples/narrow_passage_rl/tests/test_vln_safety_adapter.py

python examples/narrow_passage_rl/eval_vln_safety_adapter.py \
  --predictions examples/narrow_passage_rl/results/vln_batch_diagnostic/predictions.csv \
  --output-dir examples/narrow_passage_rl/results/vln_safety_adapter_smoke_NEW_TIMESTAMP
```

### 已有证据

- `results/vln_dataset_free/results.json`
- `results/vln_dataset_free/report.md`
- `results/vln_batch_diagnostic/metadata.json`
- `results/vln_batch_diagnostic/report.md`
- `results/vln_safety_adapter_smoke/report.md`
- `results/vln_r2r_eval_preflight/r2r_vlnce_preflight.md`
- `docs/vln_integration_status.md`

## 19. 可用于论文或汇报的准确表述

建议表述：

> 当前 NaVILA 8.49B 视觉语言模型已在 RTX A4000 上以 NF4 4-bit 方式完成
> 本地运行验证，8 帧输入下平均生成延迟约为 1.47 秒。受控诊断显示模型存在
> 前进动作先验、停止与左右方向 grounding 不足及视觉扰动敏感性，因此系统将
> NaVILA 限定为低频语义/航点提议模块，并由独立的几何可行性、失败记忆、局部
> 控制器和硬件急停承担物理执行约束。相机安装与机器人初始位姿泛化尚需通过
> 显式外参、机器人中心航点、三维域随机化和真实多视角数据进行验证。

不得表述为：

- “当前 NaVILA 已解决真实机器人 VLN”；
- “安全 adapter 提升了标准 VLN benchmark”；
- “模型已经学会物理安全停止和窄通道拒绝”；
- “当前 R2R 文本探针证明了未见场景泛化”；
- “Lite3 VLN 闭环已经具有论文级可复现结果”。

